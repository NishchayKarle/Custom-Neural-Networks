### ACCURACY with best values for nl, nh

* epocs = 200, batch size = 10, train images = 6000, test images = 10000, hidden layers = 1, neurons in hidden layer = 20
  * ACC: 91.1 %

* epocs = 200, batch size = 100, train images = 60000, test images = 10000, hidden layers = 1, neurons in hidden layer = 20 
  * ACC: 93.3 %


### Serial vs OPENP TIME TAKEN

* SERIAL: epocs = 200, batch size = 10, train images = 6000, test images = 10000, hidden layers = 1, neurons in hidden layer = 20
    * TIME: 166.8 (s)

* OMP: epocs = 200, batch size = 10, train images = 6000, test images = 10000, hidden layers = 1, neurons in hidden layer = 20
  * TIME: 33.32 (s)
  * batch size = num of threads for the loop that goes over m images

* SPEEDUP = 5.2 times


### COMPILING AND RUN

* serial: 
  * make test_nn
  * ./test_nn nl nh ne nb

* omp:
  * make test_nn_omp
  * ./test_nn_omp nl nh ne nb