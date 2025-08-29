''' Learning rate schedule
'''

import math

# -------------------------- Generic Scheduler --------------------------
class Schedule:
    '''
      Generic schedule class
    '''
    def __init__(
        self,
        start_val: int,
        end_val: int,
        T_max: int,
        step_every: int = 1
    ) -> None:
        
         step_every = int(step_every)
         assert step_every >= 1, 'step_every must be positive and greater than 0'
         self._steps = 0
         self.start_val = start_val
         self.end_val = end_val
         self.T_max = T_max # not inclusive of warmup steps
         self.step_every = step_every
         self.val = start_val
         
    def step(self, n_steps: int = 1) -> float:
        for _ in range(n_steps):
            self._steps += 1
            if self._steps % self.step_every == 0:
                self.val = self.update()
            
        return self.val

    def state_dict(self) -> dict:
        return { '_steps' : self._steps,
                 'start_val' : self.start_val,
                 'end_val' : self.end_val,
                 'T_max' : self.T_max,
                 'val': self.val,
                 'step_every' : self.step_every}
    
    def load_state_dict(self, state_dict: dict) -> None:
        self._steps = state_dict['_steps']
        self.start_val = state_dict['start_val']
        self.end_val = state_dict['end_val']
        self.T_max = state_dict['T_max']
        self.val = state_dict['val']
        self.step_every = state_dict['step_every']
    
    def reset(self) -> None:
        self._steps = 0
        self.val = self.start_val
    
    def update(self) -> None:
        raise NotImplementedError('Please implemenet update function in child class')

class WarmupCosineSchedule(Schedule):
    '''
        Cosine schedule with Linear warmup
    '''
    def __init__(
        self,
        start_val: float,
        end_val: float,
        T_max: int,
        ref_val: float,
        warmup_steps: int,
        step_every = 1,
        plateau: bool = True
    ) -> None:
        
        super().__init__(
            start_val = start_val,
            end_val = end_val,
            T_max = T_max,
            step_every = step_every
        )

        self.ref_val = ref_val
        self.warmup_steps = warmup_steps
        self.plateau = plateau

    def update(self) -> None:
        # -- stop oscillating if training continues
        if self.plateau and self._steps > self.T_max:
            val = self.end_val
        #  -- linear warmup
        elif self._steps < self.warmup_steps:
            progress = float(self._steps) / float(max(1, self.warmup_steps))
            val = self.start_val + progress * (self.ref_val - self.start_val)
        # -- cosine annealing after warmup
        else:
            progress = float(self._steps - self.warmup_steps) / float(self.T_max - self.warmup_steps)
            val = self.end_val + (self.ref_val - self.end_val ) * 0.5 * (1 + math.cos(progress * math.pi))   # cosine annealing formula

            # -- clip wd to prevent rounding errors
            if self.ref_val <= self.end_val:
                val = min(self.end_val, val)
            else:
                val = max(self.end_val, val)
        
        return val
    
    def state_dict(self) -> dict:
        return { 'ref_val' : self.ref_val,
                 'warmup_steps' : self.warmup_steps,
                 'plateau': self.plateau,
                 'schedule' : super().state_dict()}
    
    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict['schedule'])
        self.ref_val = state_dict['ref_val']
        self.warmup_steps = state_dict['warmup_steps']
        self.plateau = state_dict['plateau']
            
# -------------------------- Generic Param Scheduler --------------------------
class ParamSchedule:
    '''
    Base class for parameter schedules
    '''
    def __init__(
        self,
        schedule: Schedule
    ) -> None:
        self.schedule = schedule
        self.update(schedule.start_val)
        
    def update(self, x: float) -> None:
        raise NotImplementedError('Param Schedular Update Function Not Implemented')

    def step(self, n_steps: int = 1) -> float:
        val = self.schedule.step(n_steps)
        self.update(val)
        return val
    
    def state_dict(self) -> dict:
        return {'schedule' : self.schedule.state_dict()}
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.schedule.load_state_dict(state_dict['schedule'])
        self.update(self.schedule.val)
        
    def reset(self) -> None:
        self.schedule.reset()
        self.update(self.schedule.start_val)
             
class LRSchedule(ParamSchedule):
    '''
    Learning rate schedule 
    '''
    def __init__(
        self,
        optimiser,
        schedule: Schedule,
    ):
        self.optimiser = optimiser
        self.scale = 1.0
        self.name = 'lr'
        super().__init__(schedule)

    def update(self, lr: float):
        # -- update lr
        for group in self.optimiser.param_groups:
            if ('lr_exclude' not in group) or not group['lr_exclude']:
                ls = group.get('layer_scale', 1.0)
                group['lr'] = lr * self.scale * ls
            
    def state_dict(self) -> dict:
        return {
            'scale' : self.scale,
            'schedule': self.schedule.state_dict()
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        self.scale = state_dict['scale']
        self.schedule.load_state_dict(state_dict['schedule'])