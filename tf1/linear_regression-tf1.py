import os
import sys
import numpy as np
import argparse 
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split 
import shutil

"""
This program shows how linear regression can be done 
with tensorflow.

Here the data is simulared using numpy. 


- Jayanti Prasad [prasad.jayanti@gmail.com]


"""



def func (a, b, x):
   return a *x + b 
 
def get_data ():
  
   a, b = 6.22, 2.78
   mu, sigma  = 0.0, 1.0 
   
   np.random.seed (seed=272)
   x = np.arange(0,10, 0.05)
   x = x.reshape ([x.shape[0], 1])
   x = x / np.mean (x) 

   y = func (a, b, x)

   noise  =  np.random.normal(mu,sigma,[x.shape[0],1])
   y +=noise

   x_train, x_test, y_train, y_test \
     = train_test_split(x, y, test_size=0.2, random_state=292)
   return x_train, y_train, x_test, y_test 
 
if __name__ == "__main__":

   parser = argparse.ArgumentParser()

   parser.add_argument('-o','--output-dir',help='Output directory')
   parser.add_argument('-n','--niter',help='Number of iterations', type=int)

   args = parser.parse_args() 
 
   x_train, y_train, x_test, y_test = get_data ()
   print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
   
   os.makedirs(args.output_dir, exist_ok=True)
   log_dir = args.output_dir + os.sep + "log"

   if os.path.isdir(log_dir):
      print("Deleting log dir") 
      shutil.rmtree(log_dir)
   else :
      print("Creating log dir")
      os.makedirs(log_dir, exist_ok=True)


   print("logdir:", log_dir)

   np.random.seed (seed=293)

   A_init = (0.5 - np.random.random([1])) 
   B_init = (0.5 - np.random.random([1]))

   print("A_init:",A_init,"B_init:",B_init)

   A = tf.Variable (A_init, dtype=float)
   B = tf.Variable (B_init, dtype=float)

   x = tf.placeholder(dtype=float, shape=(None,1),name="Input")

   y = tf.multiply(A, x)  + B

   cost_function = tf.reduce_mean(tf.square(y - y_train))

   optimizer = tf.train.GradientDescentOptimizer(0.05)
   train = optimizer.minimize(cost_function)
   model = tf.initialize_all_variables()

   first_summary = tf.summary.scalar(name='loss', tensor=cost_function)


   with tf.Session() as session:
      merged = tf.summary.merge_all() 
      session.run(model)
      for step in range(0,args.niter):
         session.run(train, feed_dict={x:x_train})
         writer = tf.summary.FileWriter(log_dir,session.graph)
         loss = session.run(cost_function, feed_dict={x:x_train})
         print("step:", step, "A:",A.eval(),"B:",B.eval(),loss)
         summary = session.run(first_summary,feed_dict={x:x_train})
         writer.add_summary(summary, step)  
      y_predict = session.run (y, feed_dict={x:x_test}) 


   x_test  = x_test.reshape(x_test.shape[0]) 
   plt.scatter(x_test, y_test)
   plt.plot(x_test, y_predict,'r-')
   plt.show()

