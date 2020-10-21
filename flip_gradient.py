#coding=utf-8
import tensorflow as tf
from tensorflow.python.framework import ops


class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0
     #使实例能够像函数一样被调用
    #不设置初始值的话l默认为1，如果设置初始值，则为设置的值。
    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        '''
        @用作函数的修饰符，可以在模块或者类的定义层内对函数进行修饰
        出现在函数定义的前一行
        只可以在模块或类丁一层内对函数进行修饰
        一个修饰符就是一个函数，将被修饰的函数作为参数，在目的函数执行前，执行一些自己的操作
        
        实例：
        @dec1(arg1,arg2)
        def test(testarg)
        效果类似于#coding=utf-8
        dec1(arg1,arg2)(test(arg))
        
        '''

        '''
        定义op的梯度，梯度函数的签名为def _flip_gradients(op, grad):
        op用于接收要计算的梯度，grad用于接受上层传来的梯度
        '''
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()

        with g.gradient_override_map({"Identity": grad_name}):
            # 恒等映射的意思
            y = tf.identity(x)
        self.num_calls += 1
        return y
    
flip_gradient = FlipGradientBuilder()
