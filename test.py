import torch

def test_unsqueeze():
    # 创建一个 1 维张量
    x = torch.tensor([1, 2, 3, 4])
    print('x.dim = ', x.dim())
    print('x.shape = ', x.shape)


    x_unsqueezed0 = x.unsqueeze(0)
    print ("x_unsqueezed0 = ", x_unsqueezed0)
    print("x_unsqueezed0.shape = ", x_unsqueezed0.shape)
    print("--------")
    x_unsqueezed1 = x.unsqueeze(1)
    print ("x_unsqueezed1 = ", x_unsqueezed1)
    print("x_unsqueezed1.shape = ", x_unsqueezed1.shape)
    print("--------")

    x_unsqueezed10 = x_unsqueezed1.unsqueeze(0)
    print ("x_unsqueezed10 = ", x_unsqueezed10)
    print("x_unsqueezed10.shape = ", x_unsqueezed10.shape)
    print("--------")

    x_unsqueezed11 = x_unsqueezed1.unsqueeze(1)
    print ("x_unsqueezed11 = ", x_unsqueezed11)
    print("x_unsqueezed11.shape = ", x_unsqueezed11.shape)
    print("--------")

    x_unsqueezed12 = x_unsqueezed1.unsqueeze(2)
    print ("x_unsqueezed12 = ", x_unsqueezed12)
    print("x_unsqueezed12.shape = ", x_unsqueezed12.shape)

def test_triu():
    # 创建一个 3x3 的矩阵
    x = torch.ones((1, 10, 10))

    # 返回矩阵的上三角部分（包括主对角线）
    upper_triangular = torch.triu(x, diagonal=1).bool()

    print("原始矩阵:")
    print(x)
    print("\n上三角部分:")
    print(upper_triangular)

if __name__ == '__main__':
    test_triu()
