## 字节跳动
# 给定N种不同颜色的球以及每种颜色的球的数量，把它们放进一个容器里面，随机抓取。
# 要求写程序实现该功能，并且要按照每种颜色球的概率返回对应的球的编号

# 比如，有A, B, C 3种颜色的球，数量分别是1，2，3
# 然后把它们统一放入盒子里，随机抓取（使用random随机生成(0, 1)之间的小数）
# 按照它们各自的频数返回对应的颜色的球。


import random
def drawBalls(color_num, ball_num_list):
    """
    Parameters:
    color_num (int): Number of colors of the balls
    ball_num_list (list): Number of balls for each color

    Returns:
    int: 1-indexed color index of randomly drawn ball, as per order in ball_num_list

    """
    total_ball_num = sum(ball_num_list)
    porpotion_by_color= [1/total_ball_num * num for num in ball_num_list]
    print("porpotion_by_color", porpotion_by_color)
    rdn = random.random() # [0, 1)
    print("rdn:", rdn)
    currVal = 0
    color_idx = 0 
    for porp in porpotion_by_color:
        currVal += porp # will be 1 at the end
        color_idx += 1 # 1-index
        if rdn < currVal:
            return color_idx # will always return a value


print(drawBalls(3, [1, 2, 3]))