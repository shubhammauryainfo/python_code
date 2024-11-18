rows = int(input("enter number of rows: "))
columns = int(input("enter number of columns: "))
m = []

for i in range(rows):
    m.append([])
    for j in range(columns):
        e = int(input("enter element: "))
        m[i].append(e)

def scal(m):
    a = int(input('enter value of a: '))
    unew = []
    for i in range(rows):
        unew.append([])
        for j in range(columns):
            unew[i].append([m[i][j] * a])
    for i in range(rows):
        print(unew[i])

def tran(m):
    rmatrix = []
    rmatrix = [[0] * rows for i in range(columns)]
    for i in range(len(m)):
        for j in range(len(m[0])):
            rmatrix[j][i] = m[i][j]
    for r in rmatrix:
        print(r)

def dis(m):
    for i in range(rows):
        print("row", [i])
        print(m[i])
    for i in range(len(m[0])):
        print("column", [i])
        for j in range(len(m)):
            print("[", m[j][i], "]")

ch = True
while ch:
    print("\n1.Display matrix\n2.Display rows and columns\n3.Scalar multiplication\n4.Matrix Transpose\n5.Exit")
    ch = int(input("enter choice: "))
    if ch == 1:
        for i in range(rows):
            print(m[i])
    elif ch == 2:
        dis(m)
    elif ch == 3:
        scal(m)
    elif ch == 4:
        tran(m)
    elif ch == 5:
        print("Exit")
        ch = False
    else:
        print("\nInvalid Choice")
'''
# o/p
enter number of rows: 2
enter number of columns: 2
enter element: 4
enter element: 4
enter element: 5
enter element: 6

1.Display matrix
2.Display rows and columns
3.Scalar multiplication
4.Matrix Transpose
5.Exit
enter choice: 1
[4, 4]
[5, 6]

1.Display matrix
2.Display rows and columns
3.Scalar multiplication
4.Matrix Transpose
5.Exit
enter choice: 2
row [0]
[4, 4]
row [1]
[5, 6]
column [0]
[ 4 ]
[ 5 ]
column [1]
[ 4 ]
[ 6 ]

1.Display matrix
2.Display rows and columns
3.Scalar multiplication
4.Matrix Transpose
5.Exit
enter choice: 3
enter value of a: 3
[[12], [12]]
[[15], [18]]

1.Display matrix
2.Display rows and columns
3.Scalar multiplication
4.Matrix Transpose
5.Exit
enter choice: 4
[4, 5]
[4, 6]

1.Display matrix
2.Display rows and columns
3.Scalar multiplication
4.Matrix Transpose
5.Exit
enter choice: 4
[4, 5]
[4, 6]

1.Display matrix
2.Display rows and columns
3.Scalar multiplication
4.Matrix Transpose
5.Exit
enter choice: 5
Exit'''