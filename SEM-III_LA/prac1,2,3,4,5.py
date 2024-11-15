#PRACTICAL-1 WAP which demonstrate the following
#a)addition of two complex number
'''a=4+2j
b=3-5j
print("Addition of two complex number is:",a+b)

# o/p
 Addition of two complex number is: (7-3j)'''


#b)displaying the conjugate the complex number
'''a=4+2j
print(a.conjugate())

# o/p

(4-2j)'''


#c)plotting the a set of complex number
'''import matplotlib.pyplot as plt
x=2+2j
a=[-2+4j,-1+2j,0+2j,-1+4j,1+4j]
x=[x.real for x in a]
y=[x.imag for x in a]
plt.scatter(x,y,color="red")
plt.show()'''

#d)creating a new plot by rotating a even number by indegree 90,180,270 degreeand also by scaling by a number a=1/2,a=1/3,a=2 etc


'''import matplotlib.pyplot as plt
x=2+4j
z=1j
plt.scatter(x.real,x.imag,color='red')
c=x*2
plt.scatter(c.real,c.imag,color='blue')
plt.show()'''



#e) rotation by 180 degree
'''import matplotlib.pyplot as plt
x=2+4j
plt.scatter(x.real,x.imag,color='red')
plt.scatter(-1*x.real,-1*x.imag,color='green')
plt.show()'''



#f)rotation by 270 degree
'''import matplotlib.pyplot as plt
x=2+4j
z=-1j
plt.scatter(x.real,x.imag,color='red')
c=x*z
plt.scatter(c.real,c.imag,color='green')
plt.show()'''

#g)scaling by a=1/2,a=1/3,a=2
'''import matplotlib.pyplot as plt
x=2+4j
scale=0.5
scale1=0.33
scale2=2
plt.scatter(x.real,x.imag,color='red')
c=scale*x
d=scale1*x
e=scale2*x
plt.scatter(c.real,c.imag,color='green')
plt.scatter(d.real,d.imag,color='blue')
plt.scatter(e.real,e.imag,color='black')
plt.show()'''

#PRACTICAL-2 WAP which demonstrate the following
#a)enter a vector u as a list
#b)enter another vector v as list
#c)find the factor au+vb for different values of a&b
#d)find the dot product of u&v
'''import numpy as np
u=np.array([3,4,5])
v=np.array([1,2,7])
print("vector u",u)
print("vector v",v)
a=int(input("enter the value for a:"))
b=int(input("enter the value for b:"))
d=a*u+b*v
p=np.dot(u,v)
print("vector au+bv",d)
print("Dot product of u & v",p)

# o/p

vector u [3 4 5]
vector v [1 2 7]
enter the value for a:4
enter the value for b:5
vector au+bv [17 26 55]
Dot product of u & v 46
'''



#PRACTICAL-3
#a)basic matrix operation
'''import numpy as np
m1 = np.array([[1, 3, 4], [8, 5, 6]])
m2 = np.array([[8, 6, 9], [9, 0, 6]])
print("Add Matrix :")
a = np.add(m1, m2)
print(a)
print("Subtract Matrix")
b = np.subtract(m2, m1)
print(b)
x = np.array([[1, 7, 5], [4, 5, 3], [3, 2, 1]])
y = np.array([[6, 7, 7], [2, 3, 1], [2, 2, 3]])
t = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
print("Multiplication Matrix")
for i in range(len(x)):
    for j in range(len(y[0])):
        for k in range(len(y)):
            t[i][j] += x[i][k] * y[k][j]
for r in t:
    print(r)


#  o/p
Add Matrix :
[[ 9  9 13]
 [17  5 12]]
Subtract Matrix
[[ 7  3  5]
 [ 1 -5  0]]
Multiplication Matrix
[30 38 29]
[40 49 42]
[24 29 26]'''

#b) check if matrix is invertiable & if yes then find inverse of the matrix
'''import numpy as np
m=np.array([[1,2,1],[2,1,0],[3,0,2]])
print("matrix is:",m)
c=np.linalg.det(m)
print("determinant is:",c)
if(c!=0):
 minv=np.linalg.inv(m)
 print("inverse of matrix is:",minv)
else:
 print("matrix is not invertible")

# o/p
matrix is: [[1 2 1]
 [2 1 0]
 [3 0 2]]
determinant is: -8.999999999999998
inverse of matrix is: [[-0.22222222  0.44444444  0.11111111]
 [ 0.44444444  0.11111111 -0.22222222]
 [ 0.33333333 -0.66666667  0.33333333]]'''

#prac 4
#a)WAP toconvert a matrix into its row-echelon form
'''from sympy import *
m=Matrix([[1,0,1,3],[2,3,4,7],[-1,-3,-3,-4]])
print("matrix:{}".format(m))
M_rref=m.rref()
print("the Row echelon form of matrix m and the pivot columns:{}".format(M_rref))

# o/p
the Row echelon form of matrix m and the pivot columns:(Matrix([
[1, 0,   1,   3],
[0, 1, 2/3, 1/3],
[0, 0,   0,   0]]), (0, 1))'''

#b)WAP to find rank of a matrix
'''import numpy as np
my_matrix = np.array([[1,2,1],[3,4,7],[3,6,3]])
print("Matrix")
for row in my_matrix:
 print(row)
 rank= np.linalg.matrix_rank(my_matrix)
print("Rank of the given Matrix is :",rank)

# o/p
Matrix
[1 2 1]
[3 4 7]
[3 6 3]
Rank of the given Matrix is : 2'''


#prac5
'''import numpy as np
def opprojection(of_vec,on_vec):
 v1=np.array(of_vec)
 v2=np.array(on_vec)
 scal=np.dot(v1,v2)/np.dot(v2,v2)
 vec=scal*v2
 return round(scal,10),np.around(vec,decimals=10)
print(opprojection([4.0,4.0],[1.0,1.0]))
print(opprojection([4.0,4.0],[8.0,2.0]))


# o/p
(np.float64(4.0), array([4., 4.]))
(np.float64(0.5882352941), array([4.70588235, 1.17647059]))'''
