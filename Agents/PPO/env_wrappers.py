"""
From:  MAAC code
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from Agents.maac_utils.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            #print("Dones {}   Data {}".format(done, data))
            # try:
            if info["terminate"]:
                ob = env.reset()
            # except:
              
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'render':
            r = env.render()
            remote.send(r)
        elif cmd == 'render_human':
            r = env.render('human')
            remote.send(r)
        elif cmd == 'return_valid_act':
            r = {k:env.graph.get_valid_actions(k) for k in env.agents.keys()}
            remote.send(r)
        # elif cmd == 'reset_task':
        #     ob = env.reset_task()
        #     remote.send(ob)
        elif cmd == 'n_agents':
         #   print("in worker: {}".format(env.n_agents))
            remote.send(env.n_agents)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send([env.observation_space, env.action_space])
        # elif cmd == 'get_agent_types':
        #     if all([hasattr(a, 'adversary') for a in env.agents]):
        #         remote.send(['adversary' if a.adversary else 'agent' for a in
        #                      env.agents])
        #     else:
        #         remote.send(['agent' for _ in env.agents])
        elif cmd == "return_env":
            remote.send([env])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        # for r in self.remotes: r.send(('get_spaces', None))
        # hldr = [r.recv() for r in self.remotes]
        # print(hldr)
        # (observation_space, action_space) = zip(*hldr)
        # print(observation_space)
        # print(action_space)
        #self.remotes[0].send(('get_agent_types', None))
        #self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, rews, dones, infos  #np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes] #np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return [remote.recv() for remote in self.remotes] #np.stack([remote.recv() for remote in self.remotes])

    def render(self, indices = None):
        if indices==None:
            indices = [i for i in range(len(self.remotes))]
        
        for i,remote in enumerate(self.remotes):
            if i in indices: remote.send(('render', None))
        return [remote.recv() for i,remote in enumerate(self.remotes) if i in indices]
    
    # def render_human(self, indices = [0]):
    #     if indices==None:
    #         indices = [i for i in range(len(self.remotes))]
        
    #     for i,remote in enumerate(self.remotes):
    #         if i in indices: remote.send(('render_human', None))
    #     return [remote.recv() for i,remote in enumerate(self.remotes) if i in indices]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
    
    @property
    def n_agents(self):
        for r in self.remotes:
            r.send(("n_agents", None))
        return [r.recv() for r in self.remotes]
    
    def return_env(self):
        for r in self.remotes:
            r.send(("return_env", None))
        return [r.recv() for r in self.remotes]
    
    def return_valid_act(self):
        for r in self.remotes:
            r.send(("return_valid_act", None))
        return [r.recv() for r in self.remotes]





class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        # if all([hasattr(a, 'adversary') for a in env.agents]):
        #     self.agent_types = ['adversary' if a.adversary else 'agent' for a in
        #                         env.agents]
        # else:
        #     self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = zip(*results)#map(np.array, zip(*results))
        obs = list(obs)
        self.ts += 1
        # for (i, done) in enumerate(dones):
        #     if all(done): 
        #         obs[i] = self.envs[i].reset()
        #         self.ts[i] = 0
        for i, inf in enumerate(infos):
            if inf["terminate"]:
                obs[i] = self.envs[i].reset()
        self.actions = None
        return obs, rews, dones, infos #np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset() for env in self.envs]
        return results #np.array(results)

    def render(self, indices = None):
        results = [env.render() for env in self.envs]
        return results #np.array(results)

    def return_env(self):
        # for r in self.remotes:
        #     r.send(("return_env", None))
        return self.envs #[r.recv() for r in self.remotes]

    def return_valid_act(self):
        #for r in self.remotes:
        #    r.send(("return_valid_act", None))
        return [{k:self.envs[0].graph.get_valid_actions(k) for k in self.envs[0].agents.keys()}]
        
    @property
    def n_agents(self):
        return [e.n_agents for e in self.envs]

    def close(self):
        return