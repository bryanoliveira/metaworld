from metaworld.benchmarks.base import Benchmark
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT


class ML1(MultiClassMultiTaskEnv, Benchmark):

    def __init__(self, task_name, env_type='train', n_goals=50, sample_all=False, out_of_distribution=False, goal_seed=None):
        assert env_type == 'train' or env_type == 'test'
        
        if task_name in HARD_MODE_CLS_DICT['train']:
            cls_dict = {task_name: HARD_MODE_CLS_DICT['train'][task_name]}
            args_kwargs = {task_name: HARD_MODE_ARGS_KWARGS['train'][task_name]}
        elif task_name in HARD_MODE_CLS_DICT['test']:
            cls_dict = {task_name: HARD_MODE_CLS_DICT['test'][task_name]}
            args_kwargs = {task_name: HARD_MODE_ARGS_KWARGS['test'][task_name]}
        else:
            raise NotImplementedError

        args_kwargs[task_name]['kwargs']['random_init'] = False

        if out_of_distribution:
            # total space: low(-0.1, 0.8, 0.05), high(0.1, 0.9, 0.3)
            # train space: total space (high-low) - 20% (0.04, 0.02, 0.05) in every dimension
            # test space: remaining 20%
            if env_type == 'train':
                args_kwargs[task_name]['kwargs']['goal_low'] = (-0.1, 0.8, 0.05)
                args_kwargs[task_name]['kwargs']['goal_high'] = (0.06, 0.88, 0.25) # (0.1, 0.9, 0.3) - 20%
            else:
                args_kwargs[task_name]['kwargs']['goal_low'] = (0.06, 0.88, 0.25) # (0.1, 0.9, 0.3) - 20%
                args_kwargs[task_name]['kwargs']['goal_high'] = (0.1, 0.9, 0.3)

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)

        if goal_seed is not None:
            self.active_env.goal_space.seed(goal_seed)
        goals = self.active_env.sample_goals_(n_goals)
        self.discretize_goal_space({task_name: goals})

    @classmethod
    def available_tasks(cls):
        key_train, key_test = HARD_MODE_ARGS_KWARGS['train'], HARD_MODE_ARGS_KWARGS['test']
        tasks = sum([list(key_train)], list(key_test))
        assert len(tasks) == 50
        return tasks

    @classmethod
    def get_train_tasks(cls, task_name, sample_all=False, out_of_distribution=False, goal_seed=None):
        return cls(task_name, env_type='train', n_goals=50, sample_all=sample_all, out_of_distribution=out_of_distribution, goal_seed=goal_seed)
    
    @classmethod
    def get_test_tasks(cls, task_name, sample_all=False, out_of_distribution=False, goal_seed=None):
        return cls(task_name, env_type='test', n_goals=10, sample_all=sample_all, out_of_distribution=out_of_distribution, goal_seed=goal_seed)
