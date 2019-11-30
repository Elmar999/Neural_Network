import Neural_network as NN
import Data_manipulation as Dp


path = "iris_num.data"

dp = Dp.Data_manip(path)
dp = dp.data_matrix


nn = NN.Neural(data_matrix=dp ,batch_size = 32 , K_classes = 3 , n_hidden=1)


nn.train_epoch(n_epoch = 1000)

# print(nn.Y_train)

