extern crate densearray;
extern crate sharedmem;

use densearray::prelude::*;
use sharedmem::sync::{SpinBarrier};

use std::cmp::{min};
use std::marker::{PhantomData};
use std::sync::{Arc, Barrier, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

pub trait MemAllreduce<T, R> where R: MemReduceKernel<T> {
  fn allreduce(&self, in_buf: &[T], out_buf: &mut [T]);
}

pub trait MemMultiAllreduce<T, R> where R: MemReduceKernel<T> {
  fn multi_allreduce(&self, in_bufs: Vec<&[T]>, out_bufs: Vec<&mut [T]>);
}

pub trait MemReduceKernel<T> {
  fn commutative() -> bool { false }
  fn inc_reduce(idx: usize, src_buf: &[T], dst_buf: &mut [T]);
  fn post_reduce(_count: usize, _buf: &mut [T]) {}
}

pub trait MemMultiReduceKernel<T> {
  fn commutative() -> bool { false }
  fn check_num_bufs(src_nbufs: usize, dst_nbufs: usize) -> bool;
  fn inc_reduce(idx: usize, src_bufs: Vec<&[T]>, dst_bufs: Vec<&mut [T]>);
  fn post_reduce(_count: usize, _bufs: Vec<&mut [T]>) {}
}

pub struct SumReduce;

impl MemReduceKernel<f32> for SumReduce {
  fn commutative() -> bool { true }

  fn inc_reduce(_idx: usize, src_buf: &[f32], dst_buf: &mut [f32]) {
    dst_buf.flatten_mut().add(1.0, src_buf.flatten());
  }
}

pub struct MeanReduce;

impl MemReduceKernel<f32> for MeanReduce {
  fn inc_reduce(idx: usize, src_buf: &[f32], dst_buf: &mut [f32]) {
    let n = (idx + 1) as f32;
    dst_buf.flatten_mut().average(1.0 / n, src_buf.flatten());
  }
}

pub struct Stats2Reduce;

impl MemMultiReduceKernel<f32> for Stats2Reduce {
  fn check_num_bufs(src_nbufs: usize, dst_nbufs: usize) -> bool {
    src_nbufs == 1 && dst_nbufs == 2
  }

  fn inc_reduce(idx: usize, src_bufs: Vec<&[f32]>, dst_bufs: Vec<&mut [f32]>) {
    let n = (idx + 1) as f32;
    unimplemented!();
  }

  fn post_reduce(count: usize, bufs: Vec<&mut [f32]>) {
    unimplemented!();
  }
}

pub struct Stats4Reduce;

impl MemMultiReduceKernel<f32> for Stats4Reduce {
  fn check_num_bufs(src_nbufs: usize, dst_nbufs: usize) -> bool {
    src_nbufs == 1 && dst_nbufs == 4
  }

  fn inc_reduce(idx: usize, src_bufs: Vec<&[f32]>, dst_bufs: Vec<&mut [f32]>) {
    let n = (idx + 1) as f32;
    unimplemented!();
  }

  fn post_reduce(count: usize, bufs: Vec<&mut [f32]>) {
    unimplemented!();
  }
}

pub struct SharedRingAllreduceState<T> {
  barrier:  Arc<SpinBarrier>,
  buf_sz:   Arc<AtomicUsize>,
  parts:    Vec<Vec<Arc<Mutex<Option<(usize, Vec<T>)>>>>>,
}

pub struct SharedRingAllreduceBuilder<T> {
  num_workers:  usize,
  state:        SharedRingAllreduceState<T>,
}

impl<T> SharedRingAllreduceBuilder<T> {
  pub fn new(num_workers: usize) -> Self {
    assert!(num_workers >= 1);
    let mut parts = Vec::with_capacity(num_workers);
    for _ in 0 .. num_workers {
      let mut rank_parts = Vec::with_capacity(num_workers);
      for _ in 0 .. num_workers {
        rank_parts.push(Arc::new(Mutex::new(None)));
      }
      parts.push(rank_parts);
    }
    SharedRingAllreduceBuilder{
      num_workers:  num_workers,
      state:        SharedRingAllreduceState{
        barrier:    Arc::new(SpinBarrier::new(num_workers)),
        buf_sz:     Arc::new(AtomicUsize::new(0)),
        parts:      parts,
      },
    }
  }
}

impl<T> SharedRingAllreduceBuilder<T> where T: Clone + Default {
  pub fn into_worker<R>(self, worker_rank: usize, buf_sz: usize) -> SharedRingAllreduceWorker<T, R> where R: MemReduceKernel<T> {
    assert!(worker_rank < self.num_workers);
    let prev_buf_sz = self.state.buf_sz.compare_and_swap(0, buf_sz, Ordering::SeqCst);
    assert!(prev_buf_sz == 0 || prev_buf_sz == buf_sz);
    let max_part_sz = (buf_sz + self.num_workers - 1) / self.num_workers;
    let mut part_offset = 0;
    for p in 0 .. self.num_workers {
      let part_sz = min(max_part_sz, buf_sz - p * max_part_sz);
      let mut part_buf = Vec::with_capacity(part_sz);
      part_buf.resize(part_sz, T::default());
      let mut part = self.state.parts[worker_rank][p].lock().unwrap();
      *part = Some((part_offset, part_buf));
      part_offset += part_sz;
    }
    self.state.barrier.wait();
    SharedRingAllreduceWorker{
      worker_rank:  worker_rank,
      num_workers:  self.num_workers,
      buf_sz:       buf_sz,
      state:        self.state,
      _marker:      PhantomData,
    }
  }
}

pub struct SharedRingAllreduceWorker<T, R> where R: MemReduceKernel<T> {
  worker_rank:  usize,
  num_workers:  usize,
  buf_sz:       usize,
  state:    SharedRingAllreduceState<T>,
  _marker:  PhantomData<R>,
}

impl<T, R> MemAllreduce<T, R> for SharedRingAllreduceWorker<T, R> where T: Copy, R: MemReduceKernel<T> {
  fn allreduce(&self, in_buf: &[T], out_buf: &mut [T]) {
    assert_eq!(self.buf_sz, in_buf.len());
    assert_eq!(self.buf_sz, out_buf.len());

    if self.num_workers == 1 {
      out_buf.copy_from_slice(in_buf);
      return;
    }

    for p in 0 .. self.num_workers {
      let mut part = self.state.parts[self.worker_rank][p].lock().unwrap();
      assert!(part.is_some());
      let &mut (part_offset, ref mut part_buf) = &mut *part.as_mut().unwrap();
      let part_sz = part_buf.len();
      part_buf.copy_from_slice(&in_buf[part_offset .. part_offset + part_sz]);
    }
    self.state.barrier.wait();

    for p in 0 .. self.num_workers - 1 {
      let dst_rank = (self.worker_rank + p + 1) % self.num_workers;
      let src_part = self.state.parts[self.worker_rank][dst_rank].lock().unwrap();
      let mut dst_part = self.state.parts[dst_rank][dst_rank].lock().unwrap();
      let &(_, ref src_buf) = &*src_part.as_ref().unwrap();
      let &mut (_, ref mut dst_buf) = &mut *dst_part.as_mut().unwrap();
      R::inc_reduce(p + 1, src_buf, dst_buf);
      if !R::commutative() {
        self.state.barrier.wait();
      }
    }
    if !R::commutative() {
      let mut part = self.state.parts[self.worker_rank][self.worker_rank].lock().unwrap();
      let &mut (_, ref mut buf) = &mut *part.as_mut().unwrap();
      R::post_reduce(self.num_workers, buf);
    }
    self.state.barrier.wait();

    for p in 0 .. self.num_workers - 1 {
      let dst_rank = (self.worker_rank + p + 1) % self.num_workers;
      let src_part = self.state.parts[self.worker_rank][self.worker_rank].lock().unwrap();
      let mut dst_part = self.state.parts[dst_rank][self.worker_rank].lock().unwrap();
      let &(_, ref src_buf) = &*src_part.as_ref().unwrap();
      let &mut (_, ref mut dst_buf) = &mut *dst_part.as_mut().unwrap();
      dst_buf.copy_from_slice(src_buf);
    }
    self.state.barrier.wait();

    for p in 0 .. self.num_workers {
      let part = self.state.parts[self.worker_rank][p].lock().unwrap();
      assert!(part.is_some());
      let &(part_offset, ref part_buf) = &*part.as_ref().unwrap();
      let part_sz = part_buf.len();
      out_buf[part_offset .. part_offset + part_sz].copy_from_slice(part_buf)
    }
  }
}
