from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None # 存储 token id
        self.value = None # 存储 token id 对应的 kv_indices
        self.lock_ref = 0 # 表明当前节点存储的 token kv 正在被几个 request 使用
        self.last_access_time = time.time() # 上一次该 node 存储的 token kv 被使用的时间

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool: BaseTokenToKVPool,
        disable: bool = False,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.disable = disable
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0

    # 输入 key 是 token_ids， 返回前 n 个已经被缓存了的 kv 的 token 对应的 kv_indices
    def match_prefix(self, key: List, **kwargs):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node] # 从根节点开始搜索
        self._match_prefix_helper(self.root_node, key, value, last_node)
        if value:
            pass # 临时测试修改
            # value = torch.concat(value) # 仅仅为了测试而修改，实际 value 是 torch.tensor (里面保存着 kv_indices)
        else:
            value = torch.tensor([], dtype=torch.int32)
        return value, last_node[0]

    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def cache_finished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it finishes."""
        if token_ids is None:
            token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.disable:
            self.token_to_kv_pool.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone())
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len])

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, token_ids: Optional[List[int]] = None):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        if token_ids is None:
            token_ids = req.fill_ids

        kv_indices = self.req_to_token_pool.req_to_token[ #获取当前 request 的 kv_indices（kv在kv_pool中的位置）
            req.req_pool_idx, : len(token_ids)
        ]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(token_ids, kv_indices.clone()) #将当前 request 的 token_ids 和 kv_indices 插入到 radix cache 中, 并获取新的 prefix_len(表示前面 prefix_len 个 token 的 kv 值已经存在了)
        self.token_to_kv_pool.free(kv_indices[len(req.prefix_indices) : new_prefix_len]) # 释放掉重复的 token 的 kv

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(token_ids) # 获取 prompt_token_ids 对应的 kv_indices
        assert len(new_indices) == len(token_ids) # 因为 prompt_token_ids 已经都存入了 radix cache 中，所以这里必须向相等
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, len(req.prefix_indices) : len(new_indices)
        ] = new_indices[len(req.prefix_indices) :] # 更新该请求的 prompt 对应的 kv_indices

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)
        req.prefix_indices = new_indices
        req.last_node = new_last_node # 更新最后一个匹配上的节点

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens: int, evict_callback: Callable):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves) # heapq 是一个堆堆咧（heap_queue), 将列表 leaves 转换为一个最小堆（就地操作）, node 中定义了 __lt__ 函数，即最后一次被访问时间越早的 node 在堆的上面

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves) # 从堆中弹出最小的元素

            if x == self.root_node:
                break
            if x.lock_ref > 0: # 表明这个节点存储的 token 的 kv 正在被其他请求使用，不能删除，跳过
                continue

            evict_callback(x.value) # 这里的 evict_callback 函数，往往是来释放 kv_pool 中的 x.value（token_ids) 对应的 kv
            num_evicted += len(x.value)
            self._delete_leaf(x) # 删除该 node 及其所有的child node

            if len(x.parent.children) == 0: # 一些叶子节点被删除后，可能会产生新的叶子节点，收集这些新的叶子节点
                heapq.heappush(leaves, x.parent)

    def inc_lock_ref(self, node: TreeNode): # 将匹配到的节点 node 以及它的所有的祖先节点的 lock_ref 加一
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode): # 将匹配到的节点 node 以及它的所有的祖先节点的 lock_ref 减一
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: TreeNode
    ): # 输入 key 是 prompt_token_ids, 该函数的目的是:尽可能的找到输入 prompt_token_ids 中前 n 个已经被缓存了的 kv 的 token （姑且叫做前向最大匹配），
        # 并将前 n 已经被缓存了 kv 的 token 的 kv_indices 添加到 value 中
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        if key[0] in node.children.keys(): # 如果 prompt_token_ids 的第一个 token id 在某个 children 节点已经存在
            child = node.children[key[0]] # 获取这个 children 节点
            prefix_len = _key_match(child.key, key) #计算最大的匹配长度
            if prefix_len < len(child.key): # 如果该节点缓存的 token 长度大于 prompt_token_ids 的长度，则将这个节点进行 split，并返回新的节点
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value) # 否则即该节点缓存的 token 完全都在输入 prompt_token_ids 中
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node) # 递归继续找 prompt_token_ids 中剩下的部分（key[prefix_len:]）的 token 是否被缓存了 kv

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:][0]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len][0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        if key[0] in node.children.keys(): # 如果待插入的 token_ids 的第一个 token id 在某个 children 节点已经存在
            child = node.children[key[0]] # 获取这个 children 节点
            prefix_len = _key_match(child.key, key) #计算最大的匹配长度

            if prefix_len == len(child.key):  # 如果匹配的长度等于该节点储存的 token_id 长度，则说明该节点已经被完全匹配上，可以继续深入查找
                if prefix_len == len(key): # 待插入的 token_ids 都被匹配上了，则返回匹配的长度（查找结束）
                    return prefix_len
                else:
                    key = key[prefix_len:] # 继续递归深入查找剩余的未被匹配上的 token_ids
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            new_node = self._split_node(child.key, child, prefix_len) # 如果匹配的长度小于该节点储存的 token_id 长度，把该节点 split 掉
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:]
            )

        if len(key): # 否则则创建新的 node， 并将 token ids 和对应的 kv_indices 信息存在到 children 的 key 和 value 中
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key, f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node): # 删除 node 及其所有的child node
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self, node: TreeNode):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self): # 收集所有的叶子节点（叶子节点:该节点没有）
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list


if __name__ == "__main__":
    tree = RadixCache(None, None, False)

    prompt_token_ids = [1,2,3,4,5,6,7,8] # 假设第一个 prompt 的 token_ids 是 [1,2,3,4,5,6,7,8]
    tree.insert(prompt_token_ids)
    tree.pretty_print()

    prompt_token_ids_2 = [1,2,3,4,9,10,7,8] # 假设第二个 prompt 的 token_ids 是 [1,2,3,4,9,10,7,8]
    kv_indices, _ = tree.match_prefix(prompt_token_ids_2)
    print(f"\nkv_indices:{kv_indices}\n")
    tree.insert(prompt_token_ids_2)
    tree.pretty_print()

    def evict_callback(x):
       print("evict", x)
       return len(x)

    tree.evict(4, evict_callback)
    tree.pretty_print()
