from constraint import *

def sudoku(puzzle):
    problem=Problem()
    domain=[1,2,3,4,5,6,7,8,9]
    for i in range(9):
        for j in range(9):
            cell=(i,j)
            if puzzle[i][j]==0:
                problem.addVariable(cell, domain)
            else:
                problem.addVariable(cell, [puzzle[i][j]])
    """Adding constraints for rows and columns respectively"""
    for i in range(9): 
        rows=[(i, j) for j in range(9)]
        problem.addConstraint(AllDifferentConstraint(), rows)
    for j in range(9):
        columns=[(i, j) for i in range(9)]
        problem.addConstraint(AllDifferentConstraint(), columns)
    """Adding constrains for subgrids of sudoku puzzle"""
    for y in range (3):
        for x in range(3):
            subgrid=[(i, j) for i in range(y*3, y*3+3) for j in range(x*3, x*3+3)]
            problem.addConstraint(AllDifferentConstraint(), subgrid)
    solutions=problem.getSolutions()
    if solutions:
        solution=solutions[0]
        for i in range (9):
            row=[solution[(i,j)] for j in range (9)]
            print(" ".join(str(row[j]) if (j+1)%3!=0 or j==8 else str(row[j])+" |" for j in range(9)))
            if (i+1)%3==0 and i!=8:
                print(21*"-")
    else:
        print("No solution found.")
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]
sudoku(puzzle)