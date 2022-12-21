import numpy as np
import utilities as utl
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #anneal.txt
    x,y=utl.load_data('./datasets/anneal.txt')
    Result=utl.Result(x, y,'anneal.txt')
    print(Result)

    #diabetes.txt
    x,y=utl.load_data('./datasets/anneal.txt')
    Result=utl.Result(x, y,'diabetes.txt')
    print(Result)

    #hepatitis.txt
    x,y=utl.load_data('./datasets/hepatitis.txt')
    Result=utl.Result(x, y,'hepatitis.txt')
    print(Result)


    #kr-vs-kp.txt
    x,y=utl.load_data('./datasets/kr-vs-kp.txt')
    Result=utl.Result(x, y,'kr-vs-kp.txt')
    print(Result)

    #vote.txt
    x,y=utl.load_data('./datasets/vote.txt')
    Result=utl.Result(x, y,'vote.txt')
    print(Result)