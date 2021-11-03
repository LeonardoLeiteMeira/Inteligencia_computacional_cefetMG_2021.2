import math

# VARIAVEIS INICIAIS
x1_line = -2
sigma1 = 1
p1 = 1
q1 = 1

x2_line = 2
sigma2 = 1
p2 = 1
q2 = 1
# ----------------------
alpha = 0.1
adjusts = 4
# ----------------------

def get_w1():
    #verificar x1
    math.exp((-1/2)*(((x-x1)/sigma1)**2))

def get_w2():
    #verificar x2
    return math.exp((-1/2)*(((x-x2)/sigma2)**2))


def get_Y():
    y1 = p1*x1 + q1
    y2 = p2*x2 + q2
    w1 = get_w1()
    w2 = get_w2()
    
    y = (w1*y1 + w2*y2)/(w1+w2)
    return y;

def compare_y_value(current_y):
    print("")
    # return error

def set_new_values():
    #calcular a derivada
    #setar novos valores nas variaveis iniciais
    print("set_new_values")


#main
x = 2
yd = x**2
error = 0

for index in range (0,1000):
    current_y = get_Y()
    error = compare_y_value(current_y)
    set_new_values()

print ("Erro econtrado: "+ error)


