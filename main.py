from Houzz import Houzz


def main():
    base_folder = '/Users/chunwei/Data/houzz/'
    category = 'bedroom'
    houzz = Houzz(base_folder, category)

if __name__ == '__main__':
    main()
