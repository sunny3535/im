class A:
    def __init__(self):
        print('init')
        
    def __call__(self):
        print('call')
    
    def myfunc(self):
        print('my')
        
a = A()
a()
a.myfunc()


