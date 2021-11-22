# coding: utf-8

import os
import signal
import time
import logging
import subprocess
from threading import Thread
from multiprocessing import Process, Queue, Lock


DIAGNOSE_PERIOD = 10


class WorkerListener(object):
    def __init__(self, info_queue, err_queue):
        def wait_for_info(queue):
            while True:
                msg = queue.get()
                logging.info(msg)
        
        def wait_for_err(queue):
            while True:
                msg = queue.get()
                logging.warning(msg)
        
        self.thread1 = Thread(
            target=wait_for_info,
            args=(info_queue,)
        )
        self.thread2 = Thread(
            target=wait_for_err,
            args=(err_queue,)
        )
        self.thread1.daemon = True
        self.thread2.daemon = True
        self.thread1.start()
        self.thread2.start()


class Std2Queue(object):
    def __init__(self, queue):
        self.queue = queue

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.queue.put(line.rstrip())

    def flush(self):
        pass


class WorkerBase(Process):
    def __init__(self, id, cmd_queue, res_queues, info_queue, err_queue):
        Process.__init__(self, daemon=True)
        self.id = id
        self.cmd_queue = cmd_queue
        self.res_queues = res_queues
        self.info_queue = info_queue
        self.err_queue = err_queue

    def before_run(self):
        pass

    def do(self, **kwargs):
        return None

    def before_quit(self):
        pass

    def run(self):
        import traceback
        try:
            import sys
            sys.stdout = Std2Queue(self.info_queue)
            sys.stderr = Std2Queue(self.err_queue)
            self.before_run()

            while True:
                res_id, cmd, kwargs = self.cmd_queue.get()
                if cmd == 'exec':
                    try:
                        ret = self.do(**kwargs)
                        self.res_queues[res_id].put((0, ret))
                    except Exception as err:
                        self.res_queues[res_id].put((1, {}))
                        traceback.print_exc()
                elif cmd == 'quit':
                    self.before_quit()
                    break
        except:
            traceback.print_exc()


sub_process_pool = {}

def start_sub_proc(proc_name, id, cmd):
    global sub_process_pool
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0
    )
    if proc_name not in sub_process_pool:
        sub_process_pool[proc_name] = {}
    sub_process_pool[proc_name][id] = proc
    return proc

def kill_sub_proc(proc_name, id):
    proc = get_sub_proc(proc_name, id)
    if proc and proc.pid:
        os.kill(proc.pid, signal.SIGKILL)

def get_sub_proc(proc_name, id):
    global sub_process_pool
    if proc_name in sub_process_pool and id in sub_process_pool[proc_name]:
        return sub_process_pool[proc_name][id]
    return None

class SubWorkerBase(Thread):
    def __init__(self, id, cmd_queue, res_queues, info_queue, err_queue):
        super(SubWorkerBase, self).__init__(daemon=True)
        self.id = id
        self.pid = 9999
        self.cmd_queue = cmd_queue
        self.res_queues = res_queues
        self.info_queue = info_queue
        self.err_queue = err_queue

    def start(self):
        self.start_proc()
        return super(SubWorkerBase, self).start()

    def join(self, *args, **kwargs):
        self.kill_proc()
        return super(SubWorkerBase, self).join(*args, **kwargs)

    def start_proc_cmd(self):
        return None

    def start_proc(self):
        logging.info('start %s.' % self.__class__.__name__)
        start_sub_proc(self.__class__.__name__, self.id, self.start_proc_cmd())

    def kill_proc(self):
        logging.warning('kill %s.' % self.__class__.__name__)
        kill_sub_proc(self.__class__.__name__, self.id)

    def restart_proc(self):
        logging.warning('restart %s.' % self.__class__.__name__)
        self.kill_proc()
        self.start_proc()

    def proc_cmd(self, **kwargs):
        """ 格式化命令 """
        return None

    def proc_res(self, res):
        """ 格式化响应 """
        return None

    def before_run(self):
        pass

    def run(self):
        try:
            self.before_run()
            while True:
                res_id, cmd, kwargs = self.cmd_queue.get()
                if cmd == 'exec':
                    try:
                        proc = get_sub_proc(self.__class__.__name__, self.id)
                        pcmd = self.proc_cmd(**kwargs)
                        proc.stdin.write(pcmd)
                        proc.stdin.flush()
                        line = proc.stdout.readline().decode('utf-8')
                        if line:
                            ret = self.proc_res(line)
                            self.res_queues[res_id].put((0, ret))
                        else:
                            raise Exception()
                    except Exception as e:
                        self.res_queues[res_id].put((1, {}))
                        self.restart_proc()
                        logging.error(str(e))
                elif cmd == 'quit':
                    self.kill_proc()
                    break
        except Exception as e:
            logging.error(str(e))


class WorkerPoolBase(object):
    def __init__(self, worker_class, n_worker, n_res):
        self.n_worker = n_worker
        self.n_res = n_res
        self.apply_lock = Lock()
        self.cmd_queues = [Queue() for i in range(self.n_worker)]
        self.res_queues = [Queue() for i in range(self.n_res)]
        self.res_idle = [i for i in range(self.n_res)]

        self.info_queue = Queue()
        self.err_queue = Queue()
        self.listener = WorkerListener(self.info_queue, self.err_queue)

        self.worker_class = worker_class
        self.workers = [None for i in range(self.n_worker)]
        for i in range(self.n_worker):
            self._start(i)
        self.worker_idx = 0

        self.diagnosing = True
        self.diagnoser = Thread(target=self.__class__.diagnose, args=(self,))
        self.diagnoser.daemon = True
        self.diagnoser.start()

    def health(self, worker_id):
        return self.workers[worker_id].is_alive()

    def diagnose(self):
        while True:
            time.sleep(DIAGNOSE_PERIOD)
            if not self.diagnosing:
                break
            for w in self.workers:
                if not self.health(w.id):
                    logging.warning('restart worker[%s]: id:%d, pid:%s' % (
                        self.worker_class.__name__, w.id, w.pid
                    ))
                    self._restart(w.id)

    def _start(self, worker_id):
        w = self.worker_class(
            worker_id,
            self.cmd_queues[worker_id], self.res_queues,
            self.info_queue, self.err_queue
        )
        self.workers[worker_id] = w
        w.start()

    def _restart(self, worker_id):
        w = self.workers[worker_id]
        q = self.cmd_queues[worker_id]
        if self.health(worker_id):
            q.put((-1, 'quit', {}))
            w.join()
        self.workers[worker_id] = None
        self._start(worker_id)

    def _apply(self):
        with self.apply_lock:
            while True:
                if len(self.res_idle) > 0:
                    return self.res_idle.pop()
                else:
                    time.sleep(0.05)

    def _recycle(self, res_id):
        self.res_idle.append(res_id)

    def exec(self, **kwargs):
        res_id = self._apply()
        code, res = -1, None
        try:
            q = self.cmd_queues[self.worker_idx]
            self.worker_idx = (self.worker_idx + 1) % self.n_worker
            q.put((res_id, 'exec', kwargs))
            code, res = self.res_queues[res_id].get()
        finally:
            self._recycle(res_id)
        return code, res

    def quit(self):
        self.diagnosing = False
        for q in self.cmd_queues:
            q.put((-1, 'quit', {}))
        for w in self.workers:
            w.join()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    class TestWorker(WorkerBase):
        def do(self, **kwargs):
            print('xxoo: %s' % kwargs)
            return 0

    pool = WorkerPoolBase(TestWorker, 2, 2)
    for i in range(10):
        code, res = pool.exec(a=1, b=2, c=3)
    print(code, res)
    time.sleep(10)
    pool.quit()
