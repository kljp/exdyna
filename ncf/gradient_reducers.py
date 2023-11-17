import datetime
import os
import time
from contextlib import contextmanager
from typing import List

import numpy as np
import torch

import random
import math

class Reducer:
    def __init__(self, random_seed, device, timer):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.timer = timer

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()

class TopKReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression
        self.iteration = -1

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0

        self.iteration += 1

        with self.timer("reduce.flatpack", verbosity=2):
            flat_grad = list_to_tensor(grad_in)
            sz_grad = len(flat_grad)
            k = int(self.compression * sz_grad)

        with self.timer("reduce.topk", verbosity=2):
            _, indexes = torch.topk(flat_grad.abs(), k, sorted=False)

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                index_list = [torch.empty_like(indexes) for i in range(self.n_workers)]
                h1 = torch.distributed.all_gather(index_list, indexes, async_op=True)
                h1.wait()
                flat_indexes = torch.cat(index_list)
                unq_indexes = torch.unique(flat_indexes)
                values = flat_grad[unq_indexes.long()].contiguous()
                h2 = torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, async_op=True)
                h2.wait()
                bits_communicated = n_bits(indexes) + n_bits(values)
                params_transmitted = values.numel()
            else:
                unq_indexes = indexes
                values = flat_grad[unq_indexes.long()].contiguous()

        with self.timer("reduce.combine", verbosity=2):
            grad_temp = torch.zeros_like(flat_grad)
            grad_temp[unq_indexes.long()] = values / self.n_workers
            st_idx = 0
            for i, t in enumerate(grad_out):
                numel_t = t.numel()
                t = grad_temp[st_idx:st_idx+numel_t].reshape(t.shape)
                grad_out[i] = t
                st_idx += numel_t

        with self.timer("reduce.memory", verbosity=2):
            mem_list = flat_grad
            mem_list[unq_indexes.long()] = 0.0
            st_idx = 0
            for i, m in enumerate(memory_out):
                numel_m = m.numel()
                m = mem_list[st_idx:st_idx+numel_m].reshape(m.shape)
                memory_out[i] = m
                st_idx += numel_m

        with self.timer("reduce.printinfo", verbosity=2):
            if self.iteration % 50 == 0:
                norm_mem = torch.norm(mem_list)
                print("[Iter " + str(self.iteration) + "] [Rank " + str(int(self.rank)) + "] err=" + str(norm_mem) + ", den=" + str(params_transmitted / sz_grad))
           
        return bits_communicated, params_transmitted

class GlobalTopKReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0

        with self.timer("reduce.flatpack"):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                n = tensor.nelement()
                flatgrad_size += n
                tensor_idx.append(tensor_idx[-1] + n)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flatgrad = torch.empty(flatgrad_size, device=self.device)

            # Pack the flatgrad
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                flatgrad[start:end] = tensor.view(-1)

        top_size = max(1, int(self.compression * flatgrad.nelement()))

        with self.timer("reduce.topk", verbosity=2):
            _, positions = torch.topk(flatgrad.abs(), top_size, sorted=False)
            values = flatgrad[positions].contiguous()

        with self.timer("reduce.set_memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                local_positions = positions[(positions >= start) & (positions < end)] - start
                mem.data[:] = tensor
                mem.view(-1)[local_positions] = 0.0

        with self.timer("reduce.reduce", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, values, async_op=True)
                h2 = all_gather(worker_positions, positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [values]
                worker_positions = [positions]
            bits_communicated += n_bits(values) + n_bits(positions)
            params_transmitted += values.numel()

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0.0
                for pos, val in zip(worker_positions, worker_values):
                    local_positions = pos[(pos >= start) & (pos < end)] - start
                    local_vals = val[(pos >= start) & (pos < end)]
                    out.view(-1)[local_positions] += local_vals / self.n_workers
            
        return bits_communicated, params_transmitted

class ThreshReducer(Reducer):
     
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression
        self.iteration = -1
        self.density_actual = 0.0
        self.density_average = 0.0
        self.num_grad = 0

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0

        self.iteration += 1

        with self.timer("reduce.flatpack", verbosity=2):
            flat_grad = list_to_tensor(grad_in)
            sz_grad = len(flat_grad)

        with self.timer("reduce.threshold", verbosity=2):
            k = int(self.compression * sz_grad)
            threshold = 1.0 / (2.0 * math.sqrt(k))
            indexes, = torch.where(flat_grad.abs()>=threshold)

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                local_k = torch.tensor(indexes.numel(), device=self.device)
                ks = [torch.empty_like(local_k) for i in range(self.n_workers)]
                h1 = torch.distributed.all_gather(ks, local_k, async_op=True)
                h1.wait()
                max_sz = max(ks)
                if local_k != max_sz:
                    pad = torch.empty(max_sz - local_k, dtype=indexes.dtype, device=indexes.device)
                    indexes = torch.cat((indexes, pad), dim=0)
                index_list = [torch.empty_like(indexes) for i in range(self.n_workers)]
                h2 = torch.distributed.all_gather(index_list, indexes, async_op=True)
                h2.wait()
                indexes_temp = []
                for il, k in zip(index_list, ks):
                    if k > 0:
                        indexes_temp.append(il[:k])
                flat_indexes = list_to_tensor(indexes_temp)
                unq_indexes = torch.unique(flat_indexes)
                values = flat_grad[unq_indexes.long()].contiguous()
                h3 = torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, async_op=True)
                h3.wait()
                bits_communicated = n_bits(indexes) + n_bits(values)
                params_transmitted = values.numel()
            else:
                unq_indexes = indexes
                values = flat_grad[unq_indexes.long()].contiguous()

        with self.timer("reduce.combine", verbosity=2):
            grad_temp = torch.zeros_like(flat_grad)
            grad_temp[unq_indexes.long()] = values / self.n_workers
            st_idx = 0
            for i, t in enumerate(grad_out):
                numel_t = t.numel()
                t = grad_temp[st_idx:st_idx+numel_t].reshape(t.shape)
                grad_out[i] = t
                st_idx += numel_t

        with self.timer("reduce.memory", verbosity=2):
            mem_list = flat_grad
            mem_list[unq_indexes.long()] = 0.0
            st_idx = 0
            for i, m in enumerate(memory_out):
                numel_m = m.numel()
                m = mem_list[st_idx:st_idx+numel_m].reshape(m.shape)
                memory_out[i] = m
                st_idx += numel_m

        with self.timer("reduce.printinfo", verbosity=2):
            if self.iteration % 50 == 0:
                norm_mem = torch.norm(mem_list)
                print("[Iter " + str(self.iteration) + "] [Rank " + str(int(self.rank)) + "] err=" + str(norm_mem) + ", den=" + str(params_transmitted / sz_grad))
         
        return bits_communicated, params_transmitted

class SageReducer(Reducer):
     
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.iteration = -1
        self.alpha = 1.0
        self.beta = 1.0
        self.gamma = 1.0
        self.savg_prev = 1 # Sampled average
        self.savg_curr = 1
        self.density_ideal = compression
        self.density_actual = 0.0
        self.density_average = 0.0
        self.density_exam = 0.0 # Density examined by threshold of previous iteration
        self.samp_sz = 400
        self.tensor_sz = []
        self.num_tensor = 0
        self.num_grad = 0

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        
        self.iteration += 1

        with self.timer("reduce.flatpack", verbosity=2):
            flat_grad = list_to_tensor(grad_in)
            sz_grad = len(flat_grad)

        with self.timer("reduce.heur_params", verbosity=2):
            if self.iteration:
                with self.timer("reduce.heur_params.alpha", verbosity=2):
                    samp_acc = 0.0
                    for i in range(self.samp_sz):
                        samp_acc += abs(flat_grad[random.randrange(sz_grad)])
                    self.savg_curr = samp_acc / self.samp_sz   
                    self.alpha = self.savg_curr / self.savg_prev
                with self.timer("reduce.heur_params.beta", verbosity=2):
                    positions, = torch.where(flat_grad.abs()>=self.threshold*self.alpha)
                    numel_exam = positions.numel()
                    self.density_exam = numel_exam / sz_grad
                    exam = self.density_ideal - self.density_exam
                    self.beta = 2.0 / (1.0 + np.exp(exam))
                exam2 = self.density_exam / self.density_ideal
                if exam2 < 0.1:
                    self.gamma = 0.95
                elif exam2 > 10.0:
                    self.gamma = 1.05
                else:
                    self.gamma = 1.0
                self.threshold = self.threshold * self.alpha * self.beta * self.gamma
            else:
                self.samp_sz = int(sz_grad / (1 + sz_grad * 0.05 * 0.05)) # Slovin's formula
                samp_acc = 0.0
                for i in range(self.samp_sz):
                    samp_acc += abs(flat_grad[random.randrange(sz_grad)])
                self.savg_curr = samp_acc / self.samp_sz
                k = int(self.density_ideal * sz_grad)
                self.threshold = 1.0 / (2.0 * math.sqrt(k))


        with self.timer("reduce.threshold", verbosity=2):
            indexes, = torch.where(flat_grad.abs()>=self.threshold)

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                local_k = torch.tensor(indexes.numel(), device=self.device)
                ks = [torch.empty_like(local_k) for i in range(self.n_workers)]
                h1 = torch.distributed.all_gather(ks, local_k, async_op=True)
                h1.wait()
                max_sz = max(ks)
                if local_k != max_sz:
                    pad = torch.empty(max_sz - local_k, dtype=indexes.dtype, device=indexes.device)
                    indexes = torch.cat((indexes, pad), dim=0)
                index_list = [torch.empty_like(indexes) for i in range(self.n_workers)]
                h2 = torch.distributed.all_gather(index_list, indexes, async_op=True)
                h2.wait()
                indexes_temp = []
                for il, k in zip(index_list, ks):
                    if k > 0:
                        indexes_temp.append(il[:k])
                flat_indexes = list_to_tensor(indexes_temp)
                unq_indexes = torch.unique(flat_indexes)
                values = flat_grad[unq_indexes.long()].contiguous()
                h3 = torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, async_op=True)
                h3.wait()
                bits_communicated = n_bits(indexes) + n_bits(values)
                params_transmitted = values.numel()
            else:
                unq_indexes = indexes
                values = flat_grad[unq_indexes.long()].contiguous()

        with self.timer("reduce.combine", verbosity=2):
            grad_temp = torch.zeros_like(flat_grad)
            grad_temp[unq_indexes.long()] = values / self.n_workers
            st_idx = 0
            for i, t in enumerate(grad_out):
                numel_t = t.numel()
                t = grad_temp[st_idx:st_idx+numel_t].reshape(t.shape)
                grad_out[i] = t
                st_idx += numel_t

        with self.timer("reduce.memory", verbosity=2):
            mem_list = flat_grad
            mem_list[unq_indexes.long()] = 0.0
            st_idx = 0
            for i, m in enumerate(memory_out):
                numel_m = m.numel()
                m = mem_list[st_idx:st_idx+numel_m].reshape(m.shape)
                memory_out[i] = m
                st_idx += numel_m

        with self.timer("reduce.printinfo", verbosity=2):
            if self.iteration % 50 == 0:
                norm_mem = torch.norm(mem_list)
                print("[Iter " + str(self.iteration) + "] [Rank " + str(int(self.rank)) + "] err=" + str(norm_mem) + ", den=" + str(params_transmitted / sz_grad))

        self.savg_prev = self.savg_curr
        self.density_average = (self.iteration * self.density_average + self.density_actual) / (self.iteration + 1)
           
        return bits_communicated, params_transmitted

class DeftReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression
        self.iteration = -1

    def reduce(self, grad_in, grad_out, memory_out):
        bits_communicated = 0
        params_transmitted = 0

        self.iteration += 1

        with self.timer("reduce.flatpack", verbosity=2):
            flat_grad = list_to_tensor(grad_in)

        with self.timer("reduce.partition", verbosity=2):
            sz_grad = len(flat_grad)
            if self.iteration == 0:
                thre_part = sz_grad / self.n_workers
                self.st_layer = []
                self.end_layer = []
                alloc_pos = 0
                self.num_layer = 0
                for tensor in grad_in:
                    ts_sz = tensor.numel()
                    if ts_sz > thre_part:
                        a = int(ts_sz / self.n_workers)
                        b = int(ts_sz % self.n_workers)
                        for i in range(self.n_workers):
                            c = a
                            if b > 0:
                                c = a + 1
                                b -= 1
                            self.st_layer.append(alloc_pos)
                            alloc_pos += c
                            self.end_layer.append(alloc_pos)
                            self.num_layer += 1
                    else:
                        self.st_layer.append(alloc_pos)
                        alloc_pos += ts_sz
                        self.end_layer.append(alloc_pos)
                        self.num_layer += 1
            k = int(self.compression * sz_grad)
            norm_layer = [0 for i in range(self.num_layer)]
            for i in range(self.num_layer):
                norm_layer[i] = float(torch.norm(flat_grad[self.st_layer[i]:self.end_layer[i]]))
            priority = np.argsort(norm_layer)[::-1]
            remain_k = k
            remain_norm = sum(norm_layer)
            k_layer = [0 for i in range(self.num_layer)]
            for i in range(self.num_layer):
                pri = priority[i]
                ts_sz = self.end_layer[pri] - self.st_layer[pri]
                if remain_norm > 0:
                    temp_k = int(remain_k * (norm_layer[pri] / remain_norm))
                else:
                    temp_k = 0
                if ts_sz < temp_k:
                    k_layer[pri] = ts_sz
                else:
                    k_layer[pri] = max(1, temp_k)
                remain_k -= k_layer[pri]
                remain_norm -= norm_layer[pri]
            cycle = self.iteration % self.n_workers
            curr_part = (cycle + self.rank) % self.n_workers
            alloc_part = []
            common_part = []
            # bin-packing
            if self.rank == cycle:
                sz_bin = [0 for i in range(self.n_workers)]
                alloc_bin = [[] for i in range(self.n_workers)]
                sz_layer = [(self.end_layer[i] - self.st_layer[i]) * math.log2(k_layer[i]) if k_layer[i] > 1  else self.end_layer[i] - self.st_layer[i] for i in range(self.num_layer)]
                for i in range(self.num_layer): 
                    val_max = max(sz_layer)
                    idx_max = sz_layer.index(max(sz_layer))
                    bin_min = sz_bin.index(min(sz_bin))
                    if k_layer[idx_max] == self.end_layer[idx_max] - self.st_layer[idx_max]:
                        val_max = 0
                        common_part.append(idx_max)
                    else:
                        alloc_bin[bin_min].append(idx_max)
                    sz_bin[bin_min] += val_max
                    sz_layer[idx_max] = -1
                ts_idx = torch.tensor([len(bin) for bin in alloc_bin], dtype=torch.int32, device=self.device)
                ttl = [torch.tensor(bin, dtype=torch.int32, device=self.device) for bin in alloc_bin]
                ttl.append(torch.tensor(common_part, dtype=torch.int32, device=self.device))
                ts_val = torch.cat(ttl)
            else:
                ts_idx = torch.zeros(self.n_workers, dtype=torch.int32, device=self.device)
                ts_val = torch.zeros(self.num_layer, dtype=torch.int32, device=self.device)
            if self.n_workers > 1:
                h0_a = torch.distributed.broadcast(ts_idx, cycle, async_op=True)
                h0_b = torch.distributed.broadcast(ts_val, cycle, async_op=True)
                h0_a.wait()
                h0_b.wait()
                if self.rank == cycle:
                    alloc_part = alloc_bin[curr_part]
                else:
                    acc_pos = 0
                    for i in range(ts_idx.numel()):
                        if i == curr_part:
                            alloc_part = ts_val[acc_pos:acc_pos + int(ts_idx[i])].tolist()
                            break
                        else:
                            acc_pos += int(ts_idx[i])
                    common_part = ts_val[int(torch.sum(ts_idx)):].tolist()
            else:
                alloc_part = alloc_bin[curr_part]

        with self.timer("reduce.topk", verbosity=2):
            merged_k = 0
            if alloc_part:
                merged_ts = []
                for part in alloc_part:
                    _, part_indexes = torch.topk(flat_grad[self.st_layer[part]:self.end_layer[part]].abs(), k_layer[part], sorted=False)
                    part_indexes += self.st_layer[part]
                    merged_ts.append(part_indexes)
                    merged_k += k_layer[part]
                indexes = torch.cat(merged_ts)
            if merged_k == 0:
                _, empty_indexes = torch.topk(flat_grad[0:2].abs(), 1, sorted=False)

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                local_k = torch.tensor(merged_k, device=self.device)
                ks = [torch.empty_like(local_k) for i in range(self.n_workers)]
                h1 = torch.distributed.all_gather(ks, local_k, async_op=True)
                h1.wait()
                max_sz = max(ks)
                if local_k != max_sz:
                    if local_k > 0:
                        pad = torch.empty(max_sz - local_k, dtype=indexes.dtype, device=indexes.device)
                        indexes = torch.cat((indexes, pad), dim=0)
                    else:
                        indexes = torch.empty(max_sz, dtype=empty_indexes.dtype, device=empty_indexes.device)
                index_list = [torch.empty_like(indexes) for i in range(self.n_workers)]
                h2 = torch.distributed.all_gather(index_list, indexes, async_op=True)
                h2.wait()
                indexes_temp = []
                for il, k in zip(index_list, ks):
                    if k > 0:
                        indexes_temp.append(il[:k])
                indexes_temp.append(torch.tensor(sum([list(range(self.st_layer[part], self.end_layer[part])) for part in common_part], []), dtype=indexes.dtype, device=indexes.device))
                flat_indexes = list_to_tensor(indexes_temp)
                values = flat_grad[flat_indexes.long()].contiguous()
                h3 = torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, async_op=True)
                h3.wait()
                bits_communicated = n_bits(indexes) + n_bits(values)
                params_transmitted = values.numel()
            else:
                flat_indexes = indexes
                values = flat_grad[flat_indexes.long()].contiguous()

        with self.timer("reduce.combine", verbosity=2):
            grad_temp = torch.zeros_like(flat_grad)
            grad_avg = values / self.n_workers
            grad_temp[flat_indexes.long()] = grad_avg
            st_idx = 0
            for i, t in enumerate(grad_out):
                numel_t = t.numel()
                t = grad_temp[st_idx:st_idx+numel_t].reshape(t.shape)
                grad_out[i] = t
                st_idx += numel_t

        with self.timer("reduce.memory", verbosity=2):
            mem_list = flat_grad
            mem_list[flat_indexes.long()] = 0.0
            st_idx = 0
            for i, m in enumerate(memory_out):
                numel_m = m.numel()
                m = mem_list[st_idx:st_idx+numel_m].reshape(m.shape)
                memory_out[i] = m
                st_idx += numel_m

        with self.timer("reduce.printinfo", verbosity=2):
            if self.iteration % 50 == 0:
                norm_mem = torch.norm(mem_list)
                merged_k = 0
                merged_ts_sz = 0
                comp_complexity = 0
                for part in alloc_part:
                    merged_k += k_layer[part]
                    ts_sz = self.end_layer[part] - self.st_layer[part]
                    merged_ts_sz += ts_sz
                    if k_layer[part] > 1:
                        temp_log = math.log2(k_layer[part])
                    else:
                        temp_log = 1
                    comp_complexity += (ts_sz * temp_log)
                print("[Iter " + str(self.iteration) + "] [Rank " + str(int(self.rank)) + "] err=" + str(norm_mem) + ", loc_k=" + str(merged_k) + ", grad=" + str(merged_ts_sz) + ", complexity=" + str(comp_complexity) + ", den=" + str(params_transmitted / sz_grad))
 
        return bits_communicated, params_transmitted

class MicroReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression
        self.iteration = -1
        self.threshold = 1.0
        self.k_prev = 1
        self.acc_inefficiency = 0.0

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        
        self.iteration += 1

        with self.timer("reduce.flatpack", verbosity=2):
            flat_grad = list_to_tensor(grad_in)
            sz_grad = len(flat_grad)

        with self.timer("reduce.partition", verbosity=2):
            if self.iteration == 0:
                quotient = sz_grad // self.n_workers
                remainder = sz_grad % self.n_workers
                self.sz_part = [quotient + 1 if i in range(remainder) else quotient for i in range(self.n_workers)]
                self.sz_pos = [0 for i in range(self.n_workers)]
                for i in range(1, self.n_workers):
                    self.sz_pos[i] = self.sz_part[i - 1] + self.sz_pos[i - 1]
            cycle = self.iteration % self.n_workers
            curr_part = (cycle + self.rank) % self.n_workers
            st_part = self.sz_pos[curr_part]
            end_part = self.sz_pos[curr_part] + self.sz_part[curr_part]

        with self.timer("reduce.estimate", verbosity=2):
            k = int(self.compression * sz_grad)
            if self.iteration == 0:
                self.threshold = 1.0 / (2.0 * math.sqrt(k))
            else:
                exam = self.k_prev / k
                if exam > 1.1:
                    sf = 1.005
                elif exam > 0.909:
                    if exam > 0:
                        sf = 1.00125
                    else:
                        sf = 0.9987
                else:
                    sf = 0.995
                self.threshold = self.threshold * sf

        with self.timer("reduce.threshold", verbosity=2):
            indexes, = torch.where(flat_grad[st_part:end_part].abs()>=self.threshold)
            if len(indexes) == 0:
                indexes = torch.zeros(1, dtype=indexes.dtype, device=indexes.device)
                indexes = (indexes + self.iteration) % (end_part - st_part) + st_part
            else:
                indexes += st_part

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                local_k = torch.tensor(indexes.numel(), device=self.device)
                ks = [torch.empty_like(local_k) for i in range(self.n_workers)]
                h1 = torch.distributed.all_gather(ks, local_k, async_op=True)
                h1.wait()
                max_sz = max(ks)
                if local_k != max_sz:
                    pad = torch.empty(max_sz - local_k, dtype=indexes.dtype, device=indexes.device)
                    indexes = torch.cat((indexes, pad), dim=0)
                index_list = [torch.empty_like(indexes) for i in range(self.n_workers)]
                h2 = torch.distributed.all_gather(index_list, indexes, async_op=True)
                h2.wait()
                indexes_temp = []
                for il, k in zip(index_list, ks):
                    if k > 0:
                        indexes_temp.append(il[:k])
                flat_indexes = list_to_tensor(indexes_temp)
                values = flat_grad[flat_indexes.long()].contiguous()
                h3 = torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, async_op=True)
                h3.wait()
                bits_communicated = n_bits(indexes) + n_bits(values)
                params_transmitted = values.numel()
            else:
                flat_indexes = indexes
                values = flat_grad[flat_indexes.long()].contiguous()
            self.k_prev = params_transmitted

        with self.timer("reduce.combine", verbosity=2):
            grad_temp = torch.zeros_like(flat_grad)
            grad_temp[flat_indexes.long()] = values / self.n_workers
            st_idx = 0
            for i, t in enumerate(grad_out):
                numel_t = t.numel()
                t = grad_temp[st_idx:st_idx+numel_t].reshape(t.shape)
                grad_out[i] = t
                st_idx += numel_t

        with self.timer("reduce.memory", verbosity=2):
            mem_list = flat_grad
            mem_list[flat_indexes.long()] = 0.0
            st_idx = 0
            for i, m in enumerate(memory_out):
                numel_m = m.numel()
                m = mem_list[st_idx:st_idx+numel_m].reshape(m.shape)
                memory_out[i] = m
                st_idx += numel_m

        with self.timer("reduce.printinfo", verbosity=2):
            inefficiency = float(max_sz) * self.n_workers / self.k_prev - 1.0
            self.acc_inefficiency += inefficiency
            if self.iteration % 50 == 0:
                norm_mem = torch.norm(mem_list)
                print("[Iter " + str(self.iteration) + "] [Rank " + str(int(self.rank)) + "] err=" + str(norm_mem) + ", thre=" + str(self.threshold) + ", den=" + str(params_transmitted / sz_grad) + ", load_inc=" + str(inefficiency) + ", avg_load_inc=" + str(self.acc_inefficiency / (self.iteration + 1)))
        
        return bits_communicated, params_transmitted

class ExDynaReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression
        self.iteration = -1
        self.threshold = 1.0
        self.k_prev = 1
        self.num_blk = 1024
        self.min_blk = self.num_blk / (self.n_workers * 4) # max self.n_workers is 256
        self.acc_inefficiency = 0.0

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        
        self.iteration += 1

        with self.timer("reduce.flatpack", verbosity=2):
            flat_grad = list_to_tensor(grad_in)
            sz_grad = len(flat_grad)

        with self.timer("reduce.partition", verbosity=2):
            if self.iteration == 0:
                a = sz_grad // self.num_blk
                self.sz_blk = a - a % 32
                self.num_rem = sz_grad - self.sz_blk * self.num_blk
                quotient = self.num_blk // self.n_workers
                remainder = self.num_blk % self.n_workers
                self.blk_part = [quotient + 1 if i in range(remainder) else quotient for i in range(self.n_workers)]
                self.blk_pos = [0 for i in range(self.n_workers)]
                for i in range(1, self.n_workers):
                    self.blk_pos[i] = self.blk_part[i - 1] + self.blk_pos[i - 1]
            else:
                temp_ks = [0 for i in range(self.n_workers)]
                for i in range(self.n_workers):
                    j = ((self.iteration - 1) % self.n_workers + i) % self.n_workers
                    temp_ks[j] = self.ks_act[i]
                k_acc = int(self.k_prev / self.n_workers)
                den_acc = self.k_prev / sz_grad
                for i in range(self.n_workers - 1):
                    det = float(temp_ks[i]) / k_acc
                    det2 = float(temp_ks[i + 1]) / k_acc
                    blk_move = 1
                    if det > 1.05 and det2 < 0.95238:
                        if self.blk_part[i] - blk_move < self.min_blk:
                            continue
                        self.blk_part[i] -= blk_move
                        self.blk_part[i + 1] += blk_move
                        self.blk_pos[i + 1] -= blk_move
                        k_move = int(blk_move * self.sz_blk * den_acc)
                        temp_ks[i] -= k_move
                        temp_ks[i + 1] += k_move
                    if det < 0.95238 and det2 > 1.05:
                        if self.blk_part[i + 1] - blk_move < self.min_blk:
                            continue
                        self.blk_part[i] += blk_move
                        self.blk_part[i + 1] -= blk_move
                        self.blk_pos[i + 1] += blk_move
                        k_move = int(blk_move * self.sz_blk * den_acc)
                        temp_ks[i] += k_move
                        temp_ks[i + 1] -= k_move
            cycle = self.iteration % self.n_workers
            curr_part = (cycle + self.rank) % self.n_workers
            st_part = self.blk_pos[curr_part] * self.sz_blk
            end_part = (self.blk_pos[curr_part] + self.blk_part[curr_part]) * self.sz_blk + max(0, curr_part - self.n_workers + 2) * self.num_rem

        with self.timer("reduce.estimate", verbosity=2):
            k = int(self.compression * sz_grad)
            if self.iteration == 0:
                self.threshold = 1.0 / (2.0 * math.sqrt(k))
            else:
                exam = self.k_prev / k
                if exam > 1.1:
                    sf = 1.005
                elif exam > 0.909:
                    if exam > 0:
                        sf = 1.00125
                    else:
                        sf = 0.9987
                else:
                    sf = 0.995
                self.threshold = self.threshold * sf

        with self.timer("reduce.threshold", verbosity=2):
            indexes, = torch.where(flat_grad[st_part:end_part].abs()>=self.threshold)
            if len(indexes) == 0:
                indexes = torch.zeros(1, dtype=indexes.dtype, device=indexes.device)
                indexes = (indexes + self.iteration) % (end_part - st_part) + st_part
            else:
                indexes += st_part

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                local_k = torch.tensor(indexes.numel(), device=self.device)
                ks = [torch.empty_like(local_k) for i in range(self.n_workers)]
                h1 = torch.distributed.all_gather(ks, local_k, async_op=True)
                h1.wait()
                max_sz = max(ks)
                if local_k != max_sz:
                    pad = torch.empty(max_sz - local_k, dtype=indexes.dtype, device=indexes.device)
                    indexes = torch.cat((indexes, pad), dim=0)
                index_list = [torch.empty_like(indexes) for i in range(self.n_workers)]
                h2 = torch.distributed.all_gather(index_list, indexes, async_op=True)
                h2.wait()
                indexes_temp = []
                for il, k in zip(index_list, ks):
                    if k > 0:
                        indexes_temp.append(il[:k])
                flat_indexes = list_to_tensor(indexes_temp)
                values = flat_grad[flat_indexes.long()].contiguous()
                h3 = torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, async_op=True)
                h3.wait()
                bits_communicated = n_bits(indexes) + n_bits(values)
                params_transmitted = values.numel()
            else:
                flat_indexes = indexes
                values = flat_grad[flat_indexes.long()].contiguous()
            self.k_prev = params_transmitted
            self.ks_act = ks

        with self.timer("reduce.combine", verbosity=2):
            grad_temp = torch.zeros_like(flat_grad)
            grad_temp[flat_indexes.long()] = values / self.n_workers
            st_idx = 0
            for i, t in enumerate(grad_out):
                numel_t = t.numel()
                t = grad_temp[st_idx:st_idx+numel_t].reshape(t.shape)
                grad_out[i] = t
                st_idx += numel_t

        with self.timer("reduce.memory", verbosity=2):
            mem_list = flat_grad
            mem_list[flat_indexes.long()] = 0.0
            st_idx = 0
            for i, m in enumerate(memory_out):
                numel_m = m.numel()
                m = mem_list[st_idx:st_idx+numel_m].reshape(m.shape)
                memory_out[i] = m
                st_idx += numel_m

        with self.timer("reduce.printinfo", verbosity=2):
            inefficiency = float(max_sz) * self.n_workers / self.k_prev - 1.0
            self.acc_inefficiency += inefficiency
            if self.iteration % 50 == 0:
                norm_mem = torch.norm(mem_list)
                print("[Iter " + str(self.iteration) + "] [Rank " + str(int(self.rank)) + "] err=" + str(norm_mem) + ", thre=" + str(self.threshold) + ", den=" + str(params_transmitted / sz_grad) + ", load_inc=" + str(inefficiency) + ", avg_load_inc=" + str(self.acc_inefficiency / (self.iteration + 1)))
          
        return bits_communicated, params_transmitted

class CLTKReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression
        self.iteration = -1

    def reduce(self, grad_in, grad_out, memory_out):
        bits_communicated = 0
        params_transmitted = 0

        self.iteration += 1

        src_rank = self.iteration % self.n_workers

        with self.timer("reduce.flatpack", verbosity=2):
            flat_grad = list_to_tensor(grad_in)
            sz_grad = len(flat_grad)
            k = int(self.compression * sz_grad)

        with self.timer("reduce.topk", verbosity=2):
            _, indexes = torch.topk(flat_grad.abs(), k, sorted=False)

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                h1 = torch.distributed.broadcast(indexes, src_rank, async_op=True)
                h1.wait()
                values = flat_grad[indexes.long()].contiguous()
                h2 = torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM, async_op=True)
                h2.wait()
                bits_communicated = n_bits(indexes) + n_bits(values)
                params_transmitted = values.numel()
            else:
                values = flat_grad[indexes.long()].contiguous()

        with self.timer("reduce.combine", verbosity=2):
            grad_temp = torch.zeros_like(flat_grad)
            grad_temp[indexes.long()] = values / self.n_workers
            st_idx = 0
            for i, t in enumerate(grad_out):
                numel_t = t.numel()
                t = grad_temp[st_idx:st_idx+numel_t].reshape(t.shape)
                grad_out[i] = t
                st_idx += numel_t

        with self.timer("reduce.memory", verbosity=2):
            mem_list = flat_grad
            mem_list[indexes.long()] = 0.0
            st_idx = 0
            for i, m in enumerate(memory_out):
                numel_m = m.numel()
                m = mem_list[st_idx:st_idx+numel_m].reshape(m.shape)
                memory_out[i] = m
                st_idx += numel_m

        with self.timer("reduce.printinfo", verbosity=2):
            if self.iteration % 50 == 0:
                norm_mem = torch.norm(mem_list)
                print("[Iter " + str(self.iteration) + "] [Rank " + str(int(self.rank)) + "] err=" + str(norm_mem) + ", den=" + str(params_transmitted / sz_grad))
           
        return bits_communicated, params_transmitted

class AccordionTopKReducer(Reducer):
    """
    Modified from https://github.com/uw-mad-dash/Accordion
    """
    def __init__(self, random_seed, device, timer, k_low=0.1, k_high=0.99, detection_threshold=0.5, switch_freq=10):
        super().__init__(random_seed, device, timer)
        self.k_low = k_low
        self.k_high = k_high
        self.detection_threshold = detection_threshold
        self.switch_freq = switch_freq

    def reduce(self, grad_in, grad_out, memory_out, auto_scale_tensor, prev_norms, curr_norms, prev_lrs, curr_lrs, epoch_count):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        :auto_scale_tensor: tensor
        :prev_norms: list
        curr_norms: list
        prev_lrs:list
        curr_lrs:list
        """
        bits_communicated = 0
        params_transmitted = 0

        with self.timer("reduce.autoscale", verbosity=2):
            # Determine compression ratio for the next switch_freq epochs
            if epoch_count%self.switch_freq == 0:
                for i, grad in enumerate(grad_out):
                    curr_norms[i] = l2norm(grad)
                    if epoch_count == 0 or (prev_lrs[i] > curr_lrs[i]) or abs(prev_norms[i]-curr_norms[i])/prev_norms[i] > self.detection_threshold:
                        auto_scale_tensor[i] = self.k_high
                    else:
                        auto_scale_tensor[i] = self.k_low
                    prev_norms[i] = curr_norms[i]
                    prev_lrs[i] = curr_lrs[i]
                #Broadcast the low and high rank values from rank 0
                torch.distributed.broadcast(auto_scale_tensor, src=0)

        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for i, tensor in enumerate(grad_in):
                top_size = max(1, int(auto_scale_tensor[i].item() * tensor.nelement()))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("reduce.topk", verbosity=2):
            for i, (tensor, start, end) in enumerate(zip(grad_in, flatgrad_start_idx, flatgrad_end_idx)):
                top_size = max(1, int(auto_scale_tensor[i].item() * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                #_, indices = (tensor.view(-1).abs()).sort(descending = True)
                #positions = indices[:top_size]
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
            
        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
            
        return bits_communicated, params_transmitted


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


class ExactReducer(Reducer):
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """

        with self.timer("reduce.zero_mem", verbosity=2):
            for mem in memory_out:
                mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.reduce", verbosity=2):
            bits_communicated, params_transmitted = reduce_mean_list(self.device, list_in, list_out, self.timer)

        #print('[Rank ' + str(self.rank) + '] bits = ' + str(bits_communicated))
        return bits_communicated, params_transmitted


def reduce_mean_list(
    device: torch.device, list_in: List[torch.Tensor], list_out: List[torch.Tensor], timer
):
    if torch.distributed.is_available():
        n_workers = torch.distributed.get_world_size()
    else:
        n_workers = 1

    if n_workers == 1:
        for t_in, t_out in zip(list_in, list_out):
            t_out[:] = t_in
        return 0,0

    with timer("reduce.mean.pack"):
        buffer = TensorBuffer(list_in)

    with timer("reduce.mean.allreduce"):
        buffer.all_reduce()
        buffer.buffer /= n_workers
        bits_communicated = buffer.bits()
        params_transmitted = buffer.nelement()

    with timer("reduce.mean.unpack", verbosity=2):
        buffer.unpack(list_out)
        
    return bits_communicated, params_transmitted


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()

class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors]) # copies
    
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)
    
    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers
    

def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


@torch.jit.script
def l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))


def normalize_(tensor):
    """Divide by L2 norm. In place"""
    tensor /= l2norm(tensor)

def list_to_tensor(input_list):
    temp_list = [t.reshape(-1) for t in input_list]
    return (torch.cat(temp_list))
