#Gradient Descent 




def gradient_descent(x,y):
    m_c=b_c=0               #random assumed unknown values
    i=1000                   #number of iterations
    n=len(x)
    learningrate=0.001      #the coefficient that controls the rate of converge to the optimized state

    
    for s in range(i):
        y_p=[]
        for i in range(n):
            y_p.insert(i,m_c*x[i] + b_c)            #creating a dupe with intially assumed parameters  
        s1=0
        
        for j in range(n):
            s1+=x[j]*(y[j]-y_p[j])              #gradient of cost function
            
        md=(2/n)*s1

        
        s2=0
        for h in range(n):
            s2+=y[h]-y_p[h]
        bd=(2/n)*s2
        #Error
        S=0
        for a in range(n):
            d=(y[i]-y_p[i])**2
            S+=d
        E=S/n
        
        m_c=m_c+learningrate*md                   #prediction with a margin of error. 
        b_c=b_c+learningrate*bd
        print('m={}-b={}-Iteration={}-Error={}'.format(m_c,b_c,s,E))
        print('\n')
    print('The following relation holds between the domain range values: \ny={}x+{} with an error of {}'.format(m_c,b_c,E))
        




x=[1,2,3,4,5,6,7,8,9,10]
y=[2,3,4,5,6,7,8,9,10,11]
n=[]
for i in y:
    n.insert(y.index(i),i)

math=[92,56,88,70,80,49,65,35,66,67]
c_s=[98,68,81,80,83,52,66,30,68,73]

x=eval(input('Enter the domain list (x):'))
y=eval(input('Enter the range (y):'))


gradient_descent(x,y)




    
    
