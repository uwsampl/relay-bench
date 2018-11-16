from lstm.pytorch.sequence_models_tutorial import bm as pbm
from lstm.pytorch.test_speed import main as test_speed
from lstm.relay.pos import bm as rbm
from lstm.relay.random_lstm import main as test_relay_speed

def main():
    #rbm()
    #pbm()
    test_speed()
    test_relay_speed()

if __name__ == "__main__":
    main()
