class Hyojun_in:
    def hyojun_in():
        while True:
            a = input("image number? > ")
            try:
                b = int(a)
                if 0 <= b <= 9999:
                    break
                else:
                    print(b, "is out of bounds. please input 0~9999.")
            except ValueError:
                print("illegal input from keyboard.")
                
        return b
