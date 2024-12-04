if __name__=="__main__":
    from main2p import main
    from plot import save_plot

    for lr in [5e-6,3e-6,2e-6,5e-7]:
        print(lr)
        main(lr)
        save_plot(lr)
