from applyhistorybest import applyhistorybest as ahb
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    rasp_end2end = ''
    rasp_layerbylayer = ''
    logging.info("rasp end-to-end...")
    for opt_level in range(1, 4):
        rasp_end2end += ahb('rasp_spatial_pack', 'resnet18', 'llvm', 'rpi3b', 1, opt_level,
                     False)
    for opt_level in range(1, 4):
        rasp_end2end += ahb('rasp_spatial_pack_mobilenet', 'mobilenet', 'llvm', 'rpi3b', 1, opt_level,
                     False)
    #for opt_level in range(1, 4):
    #    rasp_end2end += ahb('rasp_spatial_pack_vgg', 'vgg16', 'llvm', 'rpi3b', 1, opt_level,
    #                 False)
    logging.info("rasp layer-by-layer...")
    for opt_level in range(1, 4):
        rasp_layerbylayer += ahb('rasp_spatial_pack', 'resnet18', 'llvm', 'rpi3b', 1, opt_level,
                     True)
    for opt_level in range(1, 4):
        rasp_layerbylayer += ahb('rasp_spatial_pack_mobilenet', 'mobilenet', 'llvm', 'rpi3b', 1, opt_level,
                     True)

    print(rasp_layerbylayer)
    with open('autotvm_rasp_lbl.csv', 'w') as layerbylayer_file:
        layerbylayer_file.write(rasp_layerbylayer)
    with open('autotvm_rasp_e2e.csv', 'w') as end2end_file:
        end2end_file.write(rasp_end2end)
    print(rasp_end2end) 

if __name__ == '__main__':
    main()
