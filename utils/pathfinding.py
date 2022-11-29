from queue import PriorityQueue
from queue import Queue
import copy
from sklearn.model_selection import ParameterGrid
import numpy as np
from tabulate import tabulate
from utils.ODM_star import Mstar_OD
from utils.ODrM_star import Mstar_ODr
import time
import heapq


class Heuristics():
    class Node():
        def __init__(self, pos, move_cost, prev_act, prev_pos):
            self.pos = pos
            self.move_cost = move_cost
            self.prev_act = prev_act
            self.prev_pos = prev_pos
        def __gt__(self, n2):
            return self.move_cost > n2.move_cost
        def __ge__(self, n2):
            return self.move_cost >= n2.move_cost
        def __lt__(self, n2):
            return self.move_cost < n2.move_cost
        def __le__(self, n2):
            return self.move_cost <= n2.move_cost
        def __eq__(self, n2):
            return self.move_cost == n2.move_cost


    def __init__(self, grid, env):
        self.grid = grid
        self.env = env
        self.block_true_dist = env.view_d
        self.x_bound, self.y_bound = grid.shape
        self.directions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]

        #Treat these object types as obstacles
        self.obstacle_types = ["obstacle", "agent"]
        self.pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1,
                        (0,0):0}
        self.cost = 1 #cost of transitioning to next node

        self.prev_pos_hldr = {i:None for i in self.env.agents.keys()}
        self.dijkstra_graphs = None

    def get_blocking_obs(self, agent_pos, goal_pos):
        '''If no path from agent to goal can be found,
            the locations of the objects in way of the shortest path
            from agent to goal is returned '''
        _, path = self.a_star_search2(agent_pos, goal_pos, ignore=["agent"])
        obs_in_way = []
        if len(path) == 0:
            _, path = self.a_star_search2(agent_pos, goal_pos, ignore=["agent", "obstacle"])
            path = path[1:-1]#remove start and end
            for p in path:
                obs_types = [ob.type for ob in self.grid[p[0], p[1]]]
                if "obstacle" in obs_types:
                    obs_in_way.append(p)
        return obs_in_way

                                    # [[0,0][0,1],[0,2]]
    def init_joint_policy_graphs(self, start, end): # 每个agent单独初始化地图，存储在self.dijkstra_graphs[i]中 i 为agent下标
        '''Performs BFS for each agent and stores the result
            in a dictionary. '''
        assert type(start) == list, Exception("start parameter has to be list")
        assert type(end) == list, Exception("end parameter has to be list")
        assert len(start) == len(end), Exception("start and end positions have to be of same length")

        self.dijkstra_graphs = {}
        for i,(p_start, p_end) in enumerate(zip(start, end)):
            assert type(p_start) == tuple and type(p_end) == tuple # 如果不是这种格式，停止并报错
            assert len(p_start) == 2 and len(p_end) == 2
           # self.dijkstra_graphs[i] = self.dijkstra_search(p_end, p_start)
            self.dijkstra_graphs[i] = self.BFS(p_end, p_start) # 整个visited 字典 ，可能是一个地图[ [x,y],node]

    def expand_position(self,agent_handle, position):
        '''Returns a list of possible next positions for an agent (ignoring other agents) '''
        #assert not self.dijkstra_search is None
        assert type(position) == tuple and len(position) == 2
        assert agent_handle in self.dijkstra_graphs
        this_graph = self.dijkstra_graphs[agent_handle]
        next_postions = [self.add_tup(position, d, return_tuple=True) for d in self.directions]
        neighbours = [n_pos for n_pos in next_postions if n_pos in this_graph] # 为什么这样得到的就是 次优 路径呢
        return neighbours
    
    def get_next_joint_policy_position(self,agent_handle, position, goal_pos=None):
        '''Returns the shortest path next position for an agent'''
       # assert not self.dijkstra_search is None
        assert type(position) == tuple and len(position) == 2
        assert agent_handle in self.dijkstra_graphs
        this_graph = self.dijkstra_graphs[agent_handle] # 获取到对应的
        assert position in this_graph
        next_postions = [self.add_tup(position, d, return_tuple=True) for d in self.directions]
        next_position_costs = {this_graph[n_pos].move_cost: n_pos for n_pos in next_postions if n_pos in this_graph}
        min_cost = min(next_position_costs.keys())
        min_cost_next_pos = next_position_costs[min_cost]
        return [min_cost_next_pos]

    def get_SIC(self, vertex):
        vertex = vertex.v
        assert type(vertex) == tuple
        assert len(vertex) == len(self.dijkstra_graphs)
        SIC = 0
        for i, pos in enumerate(vertex):
            SIC += self.dijkstra_graphs[i][pos].move_cost
        return SIC
    
    def get_shorterst_path_cost(self, agent_id, pos_tup):
        assert type(pos_tup) == tuple
        return self.dijkstra_graphs[agent_id][pos_tup].move_cost



    #def mstar_search4_OD(self, start, end, inflation = 1.0, memory_limit = None, return_time_taken=False):
    def ODmstar(self, start, end, inflation = 1.0, memory_limit = None, return_time_taken=False):
        if self.dijkstra_graphs is None:
            self.init_joint_policy_graphs(start, end)
        else:
            pass
        t2 = time.time()
        #                               一个函数方法增加位置        一个获取下一个联合位置的方法                获取最小代价路径的方法         =1.0 作用不明
        mstar = Mstar_OD(start, end, self.expand_position, self.get_next_joint_policy_position, self.get_shorterst_path_cost, inflation)
        all_actions = mstar.search(OD = True, memory_limit = memory_limit)
        time_taken = time.time() - t2
        if return_time_taken:
            return all_actions, time_taken
        else:
            return all_actions
    
    #def mstar_search4_Not_OD(self, start, end):
    def mstar_Not_OD(self, start, end):
        if self.dijkstra_graphs is None:
            self.init_joint_policy_graphs(start, end)
        else:
            pass
            #print("Joint policy graphs already present...re-using graphs")
        #t2 = time.time()
        #print("Time taken for joint policy graphs: {}".format(t2 - t_hldr1))
        mstar = Mstar_OD(start, end, self.expand_position, self.get_next_joint_policy_position, self.get_shorterst_path_cost)
        all_actions = mstar.search(OD = False)
        #print("Time taken for M star4 Not OD: {}".format(time.time() - t2))
        return all_actions


    def mstar_search_ODrMstar(self, start, end, inflation = 1.0, return_time_taken = False):
        t_hldr1 = time.time()
        if self.dijkstra_graphs is None:
            #print("Joint policy graphs not initialized. Initialzing now")
            self.init_joint_policy_graphs(start, end)
        else:
            pass
            #print("Joint policy graphs already present...re-using graphs")
        t2 = time.time()
        #print("Time taken for joint policy graphs: {}".format(t2 - t_hldr1))
        mstar = Mstar_ODr(end, self.expand_position, self.get_next_joint_policy_position, self.get_shorterst_path_cost, inflation=inflation)
        all_actions = mstar.search(tuple(start), tuple(end))
        time_taken = time.time() - t2
        #print("Time taken for ODrM*: {}".format(time.time() - t2))
        if return_time_taken:
            return all_actions, time_taken
        else:
            return all_actions

    def _m_expand(self, pos, graph, coll = None):
        '''Returns a list of tuples which is the expanded vertices '''
        pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1,
                        (0,0): 0}
        n_pos = {}
        directions = [(0,0), (0,1), (0,-1), (1,0), (-1,0)]
        for i, p in enumerate(pos):
            this_graph = graph[i]
            n_pos_hldr = []
            for d in directions:
                new_pos = tuple(self.add_tup(p, d))
                p_diff = tuple(self.add_tup(new_pos, self.mult_tup(p, -1)))
                assert p_diff in pos_act.keys()
                if new_pos in this_graph:
                    n_pos_hldr.append(new_pos)
            if i in coll:
                #Get all posible next pos
                n_pos[i] = n_pos_hldr
            else:
                #get shortest path next pos
                node_costs = [this_graph[p2].move_cost for p2 in n_pos_hldr]
                min_cost_ind = np.argmin(node_costs)
                n_node = n_pos_hldr[min_cost_ind]
                n_pos[i] = [n_node]
        
        #Create all possible vertex combinations:
        combinations = ParameterGrid(n_pos)
        all_v = [tuple([c[i] for i in n_pos.keys()]) for c in combinations]

        hldr = all_v
        actions = {i:[] for i in range(len(hldr[0]))}
        prev_v = hldr[0]
        for v in hldr[1:]:
            for i, (p, p_v) in enumerate(zip(pos, v)):
                p_diff = tuple(self.add_tup(p, self.mult_tup(p_v, -1)))
                actions[i].append(pos_act[p_diff])
            prev_v = v
        return all_v

    def BFS(self, start_pos, end_pos, ignore = ["agent"]):
        assert type(start_pos) == tuple
        assert type(end_pos) == tuple

        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:   # 对于 ignore中的类型 ：    ignore表示无视这类障碍物？
            if ig in self.obstacle_types: # 如果他是 障碍物类型之一
                ind = obstacle_types.index(ig)  # 返回 ig在obstacle_type中的下标
                del obstacle_types[ind]         # 从障碍物类型中删除

        visited = dict()    # 一个字典
        q = Queue()         # 一个队列
        start_node = self.Node(start_pos, 0, None, None) # 构建一个 node，表示开始node
        q.put(start_node) # 将初始 node 添加进队列 q 中

        while not q.empty(): # q 不空
            n = q.get()     # n为 队列 q 中的第一个进入的元素
            #assert not n.pos in visited
            if not n.pos in visited:
                visited[n.pos] = n # 通过 pos 查找 node key : pos ;value : node
                next_nodes_pos = self._get_neigbours(n.pos, obstacle_types) # 找当前 node 的 邻居 排除障碍物所在地方
                for next_n in next_nodes_pos: # 对所有邻居 ：
                    hldr = tuple(next_n[0]) # hldr师next_n这个grid_node的位置
                    if not hldr in visited: # 若这个位置没有被访问过
                        q.put(self.Node(tuple(next_n[0]), n.move_cost + 1, next_n[1], n.pos)) # 将他这个grid新建成node放入q队列中
        return visited # 当找完之后，返回所有被访问过的node 即 visited字典

        
    def a_star_search2(self,start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''

        class Node2():
            def __init__(self, pos, prev_act, prev_pos):
                self.pos = pos
                self.g = None
                self.f = None #f = g + h
                self.prev_act = prev_act
                self.prev_pos = prev_pos
            # For for python heapq: 
            def __gt__(self, n2):
                return self.g > n2.g
            def __ge__(self, n2):
                return self.g >= n2.g
            def __lt__(self, n2):
                return self.g < n2.g
            def __le__(self, n2):
                return self.g <= n2.g
            def __eq__(self, n2):
                return self.g == n2.g

        class myPriorityQ2():
            ''' Specifically for use with Node class objects'''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                self.lookup[node.pos] = node

            def pop(self):
                (_, n) = heapq.heappop(self.q)
                while not n.pos in self.lookup:
                    assert len(self.q) != 0
                    n = heapq.heappop(self.q)
                return n
 
            def __len__(self,):
                return len(self.q)
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
            def __contains__(self, other_node):
                if other_node.pos in self.lookup:
                    return True
                else:
                    return False
            def contains_less_than(self, node):
                ''' If priority Q contains same node
                    with f value less than input node.
                    If False returned, node should be 
                    added to open list'''
                if node.pos in self.lookup:
                    if self.lookup[node.pos].f <= node.f:
                        return True
                    else:
                        return False
                else:
                    return False

    
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        
        assert type(start_pos) == tuple
        assert type(goal_pos) == tuple
        def get_cost(v_current, v_next):
            return 1

        heuristic_f = self.abs_dist
        #open = myPriorityQ()
        open = myPriorityQ2()
        closed = dict()
        vs = Node2(start_pos, None, None)
        vs.g = 0
        vs.f = vs.g + heuristic_f(vs.pos, goal_pos)
        open.push((vs.f, vs))
        while not open.empty():
            vk = open.pop()
            if vk.pos == goal_pos:
                closed[vk.pos] = vk
                break #Solution found
            for vn_pos_act in self._get_neigbours(vk.pos, obstacle_types, pos_obstacle):
                (pos, act) = vn_pos_act
                pos = tuple(pos)
                vn = Node2(pos, act, vk.pos)
                vn.g = vk.g + get_cost(vk, vn)
                vn.f = vn.g + heuristic_f(vn.pos, goal_pos)
                #If vertex in open orclosed and vetex in 
                # open or closed has f vlaue less
                # than vn.f, then vn not added to open
                if open.contains_less_than(vn):
                    continue
                elif vn.pos in closed:
                    if closed[vn.pos].f <= vn.f:
                        continue
                open.push((vn.f, vn))
                # Parent already added implicity by adding action
            closed[vk.pos] = vk

        action_path = []
        node_path = []
        if goal_pos in closed.keys():
            n = closed[goal_pos]
            node_path.append(n.pos)
            while not n.prev_act is None:
                action_path.append(n.prev_act)
                n = closed[n.prev_pos]
                node_path.append(n.pos)
            action_path.reverse()
        return action_path, node_path

   

    def _get_neigbours(self, position, obstacle_types, pos_obstacle = None):
        '''pos_obstacle =  a postion to treat as obstacle '''
        assert self.assert_in_bounds(position)

        n_pos = []
        for n, a in self.pos_act.items(): # n是对坐标的加减 a是01234代表左右上下停止。
            pos = self.add_tup(position, n) # 对传入的position做加减，得到他的邻居位置pos
            if self.assert_in_bounds(pos):
                obj = self.grid[pos[0], pos[1]]
                obj_types = [o.type for o in obj] # grid是array数组  为什么用for循环，一个grid会有不同的typr吗

                if not any([obs in obj_types for obs in obstacle_types]) and pos_obstacle != (pos[0], pos[1]): # 如果这个grid 不是障碍物，并且其坐标也和障碍物不同。加入n_pos
                    n_pos.append((pos, a)) #做完动作之后的位置，以及他的动作
        return n_pos


    def assert_in_bounds(self, pos): # 判断是否超过地图边界
        (x, y) = pos
        if (x>= 0 and x<self.x_bound and y>=0 and y<self.y_bound):
            return True
        else:
            return False
    
    def abs_dist(self, curr_pos, goal_pos):
        (dx, dy) = self.add_tup(curr_pos, self.mult_tup(goal_pos, -1))
        path_len = abs(dx) + abs(dy)
        return path_len
    def abs_dist_comp(self,curr_pos, goal_pos):
        (dx, dy) = self.add_tup(curr_pos, self.mult_tup(goal_pos, -1))
        return (abs(dx),abs(dy))

    def add_tup(self, a,b, return_tuple = False):
        assert len(a) == len(b)
        ans = []
        for ia,ib in zip(a,b):
            ans.append(ia+ib)
        if return_tuple:
            ans = tuple(ans)
        return ans
    
    def mult_tup(self, a, m):
        ans = []
        for ai in a:
            ans.append(ai*m)
        return ans

    def get_valid_actions(self, agent_id):
        pos = self.env.agents[agent_id].pos
        valid_neighbours = self._get_neigbours(pos, ["obstacle", "agent"])
        if len(valid_neighbours) == 0:
            states, act = [], []
        else:
            states, act = zip(*valid_neighbours)

        act = list(act)
        states = [tuple(s) for s in states]
        if self.prev_pos_hldr[agent_id] in states:
            ind = states.index(self.prev_pos_hldr[agent_id])
            del act[ind]
        self.prev_pos_hldr[agent_id] = pos
        act.append(0)
        return act

    def is_blocking(self, agent_id):
        '''Is a particular agent blocking any other agent '''
        pos = self.env.agents[agent_id].pos
        for key, val in self.env.agents.items():
            if key == agent_id:
                continue
            goal_pos = self.env.goals[key].pos
            (dx, dy) = self.abs_dist_comp(pos, val.pos)
            if dx <= self.block_true_dist-1 and dy <= self.block_true_dist-1:
                act_path1, node_path1 = self.a_star_search2(val.pos, goal_pos)
                act_path2, node_path2 = self.a_star_search2(val.pos, goal_pos, pos_obstacle=pos)
                # if len(act_path2) != len(act_path1):
                #     print("here")
                if abs(len(act_path2) - len(act_path1)) >= (self.block_true_dist*2+1) or len(act_path2) == 0:
                    return True
        return False

    def _is_colliding(self, v):
        hldr = set()
        for i, vi in enumerate(v):
            for i2, vi2 in enumerate(v):
                if i != i2:
                    if vi == vi2:
                        hldr.add(i)
                        hldr.add(i2)
        return hldr 

    def _get_neighbours_joint(self, joint_position, obstacle_types, pos_obstacle):
        '''Takes vertex object as input and returns a 
        list of expanded v according to M* alg'''
        assert type(joint_position) == tuple
        indiv_pos = dict()
        v_len = len(joint_position)
        for i, pos in enumerate(joint_position):
            indiv_neighbours = self._get_neigbours(pos, obstacle_types, pos_obstacle)
            indiv_pos[i] = indiv_neighbours
        
        combinations = ParameterGrid(indiv_pos)
    
        all_combinations = []
        for c in combinations:
            this_joint_position = tuple([tuple(c[i][0]) for i in range(v_len)])
            this_joint_action = {i:c[i][1] for i in range(v_len)}

            if len(self._is_colliding(this_joint_position)) == 0:
                all_combinations.append([this_joint_position, this_joint_action])
        return all_combinations
            
        
    def _get_neighbours_joint_OD(self, inter_vertex, obstacle_types, pos_obstacle):
        '''Takes an intermediate vertex of the form ( (1,2, ...) , ( (x1,y1), (...),..)
        and returns a list of the next intermediate nodes. '''
        assert type(inter_vertex) == tuple
       # indiv_pos = dict()
        num_inter_levels = len(inter_vertex[1])
        current_inter_level = inter_vertex[0][-1]

        next_inter_level = current_inter_level + 1
        if next_inter_level == num_inter_levels:
            next_inter_level = 0

        joint_position = list(inter_vertex[1])

        pos_to_expand = joint_position[current_inter_level]
        neighbours = self._get_neigbours(pos_to_expand, obstacle_types, pos_obstacle)
        pos_already_assigned = joint_position[:current_inter_level]
        neighbours = [tuple(n[0]) for n in neighbours if not tuple(n[0]) in pos_already_assigned]
        #assert len(neighbours) != 0

        inter_v = []
        for n in neighbours:
            joint_p_cyp = copy.deepcopy(joint_position)
            joint_p_cyp[current_inter_level] = tuple(n)
            inter_v.append(tuple( ((next_inter_level,), tuple(joint_p_cyp))))

        return inter_v

    def a_star_search5(self, start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''


        def abs_dist_SIC(start, end):
            assert type(start) == tuple
            assert type(end) == tuple
            total= 0
            for i, (p1,p2) in enumerate(zip(start, end)):
                total += self.abs_dist(p1, p2)
            return total

        class Node2():
            def __init__(self, pos, prev_act, prev_pos):
                self.pos = pos
                self.v = None
                self.g = 1e6 #None
                self.f = None #f = g + h
                self.prev_act = prev_act
                self.prev_pos = prev_pos
                self.back_ptr = None

            def add_parent(self, v):
                self.back_ptr = v

            # For python heapq: 
            def __gt__(self, n2):
                return self.g > n2.g
            def __ge__(self, n2):
                return self.g >= n2.g
            def __lt__(self, n2):
                return self.g < n2.g
            def __le__(self, n2):
                return self.g <= n2.g
            def __eq__(self, n2):
                return self.g == n2.g

        class AllVertex():
            '''Keeps track of all nodes created
                such that nodes are created only once '''
            def __init__(self, use_intermediate_nodes = False):
                self.all_v = dict()
                self.intermediate = use_intermediate_nodes
            def get(self, v_pos):
                if v_pos in self.all_v:
                    v = self.all_v[v_pos]
                else:
                    if self.intermediate:
                        (inter_level, pos) = v_pos
                        v = Node2(pos, None, None)
                        v.v = v_pos
                    else:
                        v = Node2(v_pos, None, None)
                    self.all_v[v_pos] = v
                return v
            def update(self, v, prev_act, prev_pos):
                assert v.v in self.all_v
                v.prev_act = prev_act
                v.prev_pos = prev_pos

            def add_parent(self, v_current, v_parent):
                v_current.add_parent(v_parent)

        class simplePriorityQ():
            def __init__(self):
                self.q = []
            def push(self, item):
                heapq.heappush(self.q, item)
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                return n
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
        
        class ReplacePriorityQ():
            '''Replaces same nodes in q by only poping the lowest f-valued node '''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                self.lookup[node.v] = node
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                while not n.v in self.lookup:
                    assert len(self.q) != 0
                    n = heapq.heappop(self.q)
                del self.lookup[n.v]
                return n
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
    
            
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        assert len(start_pos) == len(goal_pos)
        assert type(start_pos) == tuple
        assert type(goal_pos) == tuple
        #Assert input is tuple of tuples:
        for hldr1, hldr2 in zip(start_pos, goal_pos):
            assert type(hldr1) == tuple
            assert type(hldr2) == tuple
        N_AGENTS = len(start_pos)

        def get_cost(v_current, v_next, intermediate = False):
            if intermediate:
                inter_level = v_next.v[0][-1]
                return 1 #1*(inter_level+1)
            else:
                return N_AGENTS

        heuristic_f = abs_dist_SIC
        #open = simplePriorityQ() 
        open = ReplacePriorityQ()
        all_v = AllVertex(use_intermediate_nodes=True)
        vs = all_v.get(((0,),start_pos))
        vs.g = 0
        vs.f = vs.g + heuristic_f(vs.pos, goal_pos)
        open.push((vs.f, vs))
        solution_v = None
        while not open.empty():
            vk = open.pop()
            if vk.pos == goal_pos and vk.v[0][-1] == 0:
                solution_v = vk
                break #Solution found
            #for vn_pos_act in self._get_neigbours(vk.pos, obstacle_types, pos_obstacle):
            for vn_pos_act in self._get_neighbours_joint_OD(vk.v, obstacle_types, pos_obstacle):
                v = vn_pos_act
                #pos = tuple(pos)
                vn = all_v.get(v)
                if vk.g + get_cost(vk, vn) < vn.g:
                    #all_v.update(vn, act, vk.v)
                    all_v.add_parent(vn, vk)
                    vn.g = vk.g + get_cost(vk, vn, intermediate=True)
                    vn.f = vn.g + heuristic_f(vn.pos, goal_pos)
                    open.push((vn.f, vn))
        if solution_v is None:
            return None
        else:
            return self._back_track(solution_v) #action_path, node_path

    def _back_track(self, goal_v):
        '''Returns a dictionary of actions for the optimal path '''
        self.pos_act = {(0,1):2,
                        (1,0):3,
                        (0,-1):4,
                        (-1,0):1,
                        (0,0): 0}
        
        #get vertices:
        all_v = []
        all_v.append(goal_v.pos)
        next_v = goal_v.back_ptr
        while not next_v is None:
            if next_v.v[0][-1] == 0:
                all_v.append(next_v.pos)
            next_v = next_v.back_ptr

        #Get actions from vertices:
        all_actions = []
        prev_v = all_v[-1]
        for v in reversed(all_v[:-1]):
            actions = {}
            for i, (previous_position, next_postion) in enumerate(zip(prev_v, v)):
                position_diff = self._add_tup(next_postion, self._mult_tup(previous_position, -1))
                actions[i] = self.pos_act[position_diff]
            prev_v = v
            all_actions.append(actions)
        return all_actions


    def _add_tup(self, a,b):
        assert len(a) == len(b)
        ans = []
        for ia,ib in zip(a,b):
            ans.append(ia+ib)
        return tuple(ans)

    def _mult_tup(self, a, m):
        ans = []
        for ai in a:
            ans.append(ai*m)
        return tuple(ans)


    def a_star_search4(self,start_pos, goal_pos, ignore = ["agent"], pos_obstacle = None):
        '''Single agent, single goal, ignore object of type agent.
            Automatically ignore goals which are not this agent's goal '''

        class Node2():
            def __init__(self, pos, prev_act, prev_pos):
                self.pos = pos
                self.g = 1e6 #None
                self.f = None #f = g + h
                self.prev_act = prev_act
                self.prev_pos = prev_pos
            # For python heapq: 
            def __gt__(self, n2):
                return self.g > n2.g
            def __ge__(self, n2):
                return self.g >= n2.g
            def __lt__(self, n2):
                return self.g < n2.g
            def __le__(self, n2):
                return self.g <= n2.g
            def __eq__(self, n2):
                return self.g == n2.g

        class AllVertex():
            '''Keeps track of all nodes created
                such that nodes are created only once '''
            def __init__(self):
                self.all_v = dict()
            def get(self, pos):
                if pos in self.all_v:
                    v = self.all_v[pos]
                else:
                    v = Node2(pos, None, None)
                    self.all_v[pos] = v
                return v
            def update(self, v, prev_act, prev_pos):
                assert v.pos in self.all_v
                v.prev_act = prev_act
                v.prev_pos = prev_pos

        class simplePriorityQ():
            def __init__(self):
                self.q = []
            def push(self, item):
                heapq.heappush(self.q, item)
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                return n
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
        
        class ReplacePriorityQ():
            '''Replaces same nodes in q by only poping the lowest f-valued node '''
            def __init__(self):
                self.q = []
                self.lookup = dict()
            def push(self, item):
                heapq.heappush(self.q, item)
                node = item[-1]
                self.lookup[node.pos] = node
            def pop(self):
                (_, n) = heapq.heappop(self.q)
                while not n.pos in self.lookup:
                    assert len(self.q) != 0
                    n = heapq.heappop(self.q)
                del self.lookup[n.pos]
                return n
            def empty(self):
                if len(self.q) == 0:
                    return True
                else:
                    return False
    
            
        obstacle_types = copy.deepcopy(self.obstacle_types)
        for ig in ignore:
            if ig in self.obstacle_types:
                ind = obstacle_types.index(ig)
                del obstacle_types[ind]
        
        #######################
        assert type(start_pos) == tuple
        assert type(goal_pos) == tuple
        def get_cost(v_current, v_next):
            return 1

        heuristic_f = self.abs_dist
        open = simplePriorityQ() 
        all_v = AllVertex()
        vs = all_v.get(start_pos)
        vs.g = 0
        vs.f = vs.g + heuristic_f(vs.pos, goal_pos)
        open.push((vs.f, vs))
        while not open.empty():
            vk = open.pop()
            if vk.pos == goal_pos:
                break #Solution found
            for vn_pos_act in self._get_neigbours(vk.pos, obstacle_types, pos_obstacle):
                (pos, act) = vn_pos_act
                pos = tuple(pos)
                vn = all_v.get(pos)
                if vk.g + get_cost(vk, vn) < vn.g:
                    all_v.update(vn, act, vk.pos)
                    vn.g = vk.g + get_cost(vk, vn)
                    vn.f = vn.g + heuristic_f(vn.pos, goal_pos)
                    open.push((vn.f, vn))
        action_path = []
        node_path = []
        closed = all_v.all_v
        if goal_pos in closed.keys():
            n = closed[goal_pos]
            node_path.append(n.pos)
            while not n.prev_act is None:
                action_path.append(n.prev_act)
                n = closed[n.prev_pos]
                node_path.append(n.pos)
            action_path.reverse()
        return action_path, node_path








