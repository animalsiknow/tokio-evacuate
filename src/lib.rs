// Copyright (c) 2018 Nuclear Furnace
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! `tokio-evacuate` provides a way to safely "evacuate" users of a resource before forcefully
//! removing them.
//!
//! In many networked applications, there comes a time when the server must shutdown or reload, and
//! may still be actively serving traffic.  Listeners or publishers can be shut down, and remaining
//! work can be processed while no new work is allowed.. but this may take longer than the operator
//! is comfortable with.
//!
//! `Evacuate` is a middleware future, that works in conjuction with a classic "shutdown signal."
//! By combining a way to track the number of current users, as well as a way to fire a global
//! timeout, we allow applications to provide soft shutdown capabilities, giving work a chance to
//! complete, before forcefully stopping computation.
//!
//! `tokio-evacuate` depends on Tokio facilities, and so will not work on other futures executors.

use futures::{
    future::{Fuse, FusedFuture, FutureExt},
    ready,
};
use parking_lot::Mutex;
use slab::Slab;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio_executor::{DefaultExecutor, Executor, SpawnError};
use tokio_sync::AtomicWaker;
use tokio_timer::{clock::now as clock_now, delay, Delay};

#[derive(Default)]
struct Inner {
    count: AtomicUsize,
    finished: AtomicBool,
    notifier: AtomicWaker,
    waiters: Mutex<Slab<Arc<AtomicWaker>>>,
}

/// Dispatcher for user count updates.
///
/// [`Warden`] is cloneable.
#[derive(Clone)]
pub struct Warden {
    state: Arc<Inner>,
}

/// A future for safely "evacuating" a resource that is used by multiple parties.
///
/// [`Evacuate`] tracks a tripwire, the count of concurrent users, and an evacuation timeout, and
/// functions in a two-step process: we must be tripped, and then we race to the timeout.
///
/// Until the tripwire completes, [`Evacuate`] will always return `Async::NotReady`.  Once we detect
/// that the tripwire has completed, however, we immediately spawn a timeout, based on the
/// configured value, and race between the user count dropping to zero and the timeout firing.
///
/// The user count is updated by calls to [`Warden::increment`] and [`Warden::decrement`].
///
/// [`Evacuate`] can be cloned, and all clones will become ready at the same time.
pub struct Evacuate {
    state: Arc<Inner>,
    waker: Arc<AtomicWaker>,
    waiter_id: usize,
}

pub struct Runner<F: Future> {
    state: Arc<Inner>,
    tripwire: Fuse<F>,
    timeout_ms: u64,
    timeout: Delay,
}

impl Inner {
    pub fn new() -> Arc<Inner> { Arc::new(Default::default()) }

    pub fn increment(&self) { self.count.fetch_add(1, Ordering::SeqCst); }

    pub fn decrement(&self) {
        if self.count.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.notifier.wake();
        }
    }

    pub fn register(&self, waiter: Arc<AtomicWaker>) -> usize {
        let mut waiters = self.waiters.lock();
        waiters.insert(waiter)
    }

    pub fn unregister(&self, waiter_id: usize) {
        let mut waiters = self.waiters.lock();
        let _ = waiters.remove(waiter_id);
    }

    pub fn wake(&self) {
        self.finished.store(true, Ordering::SeqCst);

        let waiters = self.waiters.lock();
        for waiter in waiters.iter() {
            waiter.1.wake();
        }
    }
}

impl Warden {
    /// Increments the user count.
    pub fn increment(&self) { self.state.increment(); }

    /// Decrements the user count.
    pub fn decrement(&self) { self.state.decrement(); }
}

impl<F: Future> Runner<F> {
    pub(crate) fn new(tripwire: F, timeout_ms: u64, state: Arc<Inner>) -> Runner<F> {
        Runner {
            state,
            tripwire: tripwire.fuse(),
            timeout_ms,
            timeout: delay(clock_now()),
        }
    }

    fn tripwire(self: Pin<&mut Self>) -> Pin<&mut Fuse<F>> {
        // Safety: self.tripwire is never moved.
        unsafe { self.map_unchecked_mut(|runner| &mut runner.tripwire) }
    }

    fn timeout(self: Pin<&mut Self>) -> &mut Delay {
        // Safety: self.timeout is not structurally pinned.
        unsafe { &mut self.get_unchecked_mut().timeout }
    }
}

impl<F: Future> Future for Runner<F> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        self.state.notifier.register(cx.waker().clone());

        // We have to wait for our tripwire.
        if !self.as_mut().tripwire().is_terminated() {
            let _ = ready!(self.as_mut().tripwire().poll(cx));

            // If we're here, reset our delay based on the timeout.
            let timeout_ms = self.timeout_ms;
            self.as_mut()
                .timeout()
                .reset(clock_now() + Duration::from_millis(timeout_ms));
        }

        // We've tripped, so let's see what we're at for count.  If we're at zero, then we're done,
        // otherwise, fall through and see if we've hit our delay yet.
        if self.state.count.load(Ordering::SeqCst) == 0 {
            // We've tripped and we're at count 0, so we're done.  Notify waiters.
            self.state.wake();
            return Poll::Ready(());
        }

        // Our count isn't at zero, but let's see if we've timed out yet.
        ready!(self.as_mut().timeout().poll_unpin(cx));

        // We timed out, so mark ourselves finished and notify.
        self.state.wake();

        Poll::Ready(())
    }
}

impl Evacuate {
    /// Creates a new [`Evacuate`].
    ///
    /// The given `tripwire` is used, and the internal timeout is set to the value of `timeout_ms`.
    ///
    /// Returns a [`Warden`] handle, used for incrementing and decrementing the user count, an
    /// [`Evacuate`] future, which callers can select/poll against directly, and a [`Runner`]
    /// future which must be spawned manually to drive the inner behavior of [`Evacuate`].
    ///
    /// If you're using Tokio, you can call [`default_executor`] to spawn the runner on the default
    /// executor.
    pub fn new<F>(tripwire: F, timeout_ms: u64) -> (Warden, Evacuate, Runner<F>)
    where
        F: Future + Send + 'static,
    {
        let state = Inner::new();
        let warden = Warden { state: state.clone() };

        let task = Arc::new(AtomicWaker::new());
        let waiter_id = state.register(task.clone());

        let evacuate = Evacuate {
            state: state.clone(),
            waker: task,
            waiter_id,
        };

        let runner = Runner::new(tripwire, timeout_ms, state);

        (warden, evacuate, runner)
    }

    /// Creates a new [`Evacuate`], based on the default executor.
    ///
    /// The given `tripwire` is used, and the internal timeout is set to the value of `timeout_ms`.
    ///
    /// Returns a [`Warden`] handle, used for incrementing and decrementing the user count, and an
    /// [`Evacuate`] future, which callers can select/poll against directly.
    ///
    /// This functions spawns a background task on the default executor which drives the state
    /// machine powering [`Evacuate`].  This function must be called from within a running task.
    pub fn default_executor<F>(tripwire: F, timeout_ms: u64) -> Result<(Warden, Evacuate), SpawnError>
    where
        F: Future + Send + Sized + 'static,
    {
        let (warden, evacuate, runner) = Self::new(tripwire, timeout_ms);
        let runner: Box<dyn Future<Output = ()> + Send> = Box::new(runner);

        DefaultExecutor::current()
            .spawn(runner.into())
            .map(move |_| (warden, evacuate))
    }
}

impl Drop for Evacuate {
    fn drop(&mut self) { self.state.unregister(self.waiter_id); }
}

impl Clone for Evacuate {
    fn clone(&self) -> Self {
        let state = self.state.clone();
        let task = Arc::new(AtomicWaker::new());
        let waiter_id = state.register(task.clone());

        Evacuate {
            state,
            waker: task,
            waiter_id,
        }
    }
}

impl Future for Evacuate {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        self.waker.register(cx.waker().clone());

        if !self.state.finished.load(Ordering::SeqCst) {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Evacuate;

    use futures::{
        future::{pending, ready},
        pin_mut,
        task::Context,
    };
    use std::{future::Future, time::Duration};
    use tokio_test::{self, assert_pending, assert_ready, clock, task};

    fn mock(f: impl Fn(&mut clock::Handle, &mut Context)) {
        task::mock(|cx| {
            clock::mock(|cl| f(cl, cx));
        });
    }

    #[test]
    fn test_evacuate_stops_at_tripwire() {
        mock(|_, cx| {
            let tripwire = pending::<()>();
            let (_warden, evacuate, _runner) = Evacuate::new(tripwire, 10000);
            pin_mut!(evacuate);

            assert_pending!(evacuate.as_mut().poll(cx));
        });
    }

    #[test]
    fn test_evacuate_falls_through_on_tripwire() {
        mock(|_, cx| {
            let tripwire = ready(());
            let (_warden, evacuate, runner) = Evacuate::new(tripwire, 10000);
            pin_mut!(evacuate);
            pin_mut!(runner);

            assert_pending!(evacuate.as_mut().poll(cx));
            assert_ready!(runner.as_mut().poll(cx));
            assert_ready!(evacuate.as_mut().poll(cx));
        });
    }

    #[test]
    fn test_evacuate_stops_after_tripping_with_clients() {
        mock(|_, cx| {
            let tripwire = ready(());
            let (warden, evacuate, runner) = Evacuate::new(tripwire, 10000);
            pin_mut!(evacuate);
            pin_mut!(runner);

            assert_pending!(evacuate.as_mut().poll(cx));
            warden.increment();
            assert_pending!(runner.as_mut().poll(cx));
            assert_pending!(evacuate.as_mut().poll(cx));
        });
    }

    #[test]
    fn test_evacuate_completes_after_client_count_ping_pong() {
        mock(|_, cx| {
            let tripwire = ready(());
            let (warden, evacuate, runner) = Evacuate::new(tripwire, 10000);
            pin_mut!(evacuate);
            pin_mut!(runner);

            warden.increment();
            assert_pending!(runner.as_mut().poll(cx));
            assert_pending!(evacuate.as_mut().poll(cx));
            warden.increment();
            assert_pending!(runner.as_mut().poll(cx));
            assert_pending!(evacuate.as_mut().poll(cx));
            warden.decrement();
            warden.decrement();
            assert_ready!(runner.as_mut().poll(cx));
            assert_ready!(evacuate.as_mut().poll(cx));
        });
    }

    #[test]
    fn test_evacuate_delay_before_clients_hit_zero() {
        mock(|cl, cx| {
            let tripwire = ready(());
            let (warden, evacuate, runner) = Evacuate::new(tripwire, 10000);
            pin_mut!(evacuate);
            pin_mut!(runner);

            warden.increment();
            assert_pending!(runner.as_mut().poll(cx));
            assert_pending!(evacuate.as_mut().poll(cx));
            warden.increment();
            assert_pending!(runner.as_mut().poll(cx));
            assert_pending!(evacuate.as_mut().poll(cx));
            warden.decrement();
            assert_pending!(runner.as_mut().poll(cx));
            assert_pending!(evacuate.as_mut().poll(cx));
            cl.advance(Duration::from_millis(10001));
            assert_ready!(runner.as_mut().poll(cx));
            assert_ready!(evacuate.as_mut().poll(cx));
        });
    }
}
