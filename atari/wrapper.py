import numpy as np
import scipy as sc
import scipy.misc

class FrameStack:
    """
    Fixed capacity stack-like data structure for state frames. Older frames (smaller indices) are removed
    to make way for new frames (larger indices)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.frames = []
        
    def clear(self):
        self.frames = []
        
    def push(self, frame):
        if len(self.frames) == 0:
            # If first, copy over all slots
            for i in range(self.capacity):
                self.frames.append(frame)
        else:
            # Remove older frame and add new frame
            self.frames.pop(0)
            self.frames.append(frame)
            
    def get(self):
        if len(self.frames) != self.capacity:
            raise BufferError("Not enough frames in stack")
        return np.array(self.frames)
    
class GymWrapper:
    def __init__(self, env, action_repeat=1, stack_length=4):
        """
        Creates a new wrapper around the OpenAI environment
        :param env: The OpenAI environment to wrap around
        :param action_repeat: The number of frames to apply each selected action for
        :param stack_length: The number of frames to stack as a single observation for the network input
        """
        self.env = env
        self.action_repeat = action_repeat
        self.stack_length = stack_length
        self.action_count = self.env.action_space.n
        
        self.shape = (95, 65)
        self.frame_stack = FrameStack(stack_length)

    def reset(self):
        """
        Resets the environment
        :return: preprocessed first state of next game
        """
        self.frame_stack.clear()
        state = self.preprocess(self.env.reset())
        self.frame_stack.push(state)
        return self.frame_stack.get()
    
    def preprocess(self, state):
        # Grayscale
        state = state.mean(axis=2)
        # Crop
        state = state[30:220, 15:145]
        # Resize
        state = sc.misc.imresize(state, self.shape)
        # Normalize (1. / 255)
        state = state * 0.0039215686274
        # Required by CNTK
        state = state.astype(np.float32)
        
        return state

    def step(self, action):
        """
        Executes action on the next 'action_repeat' frames
        :param action: The action to execute
        :return: (s, r, done, info): next state, reward, terminated, debugging information
        """
        rewards = 0
        for _ in range(self.action_repeat):
            s, r, done, info = self.env.step(action)
            self.frame_stack.push(self.preprocess(s))
            rewards += r
            if done:
                break
        return self.frame_stack.get(), rewards / 100.0, done, info

    def random_action(self):
        """        
        Returns a random action to execute
        """
        return self.env.action_space.sample()

    def render(self):
        """
        Renders the current state
        """
        self.env.render()

    def close(self):
        """
        Closes the rendering window
        """
        self.env.close()