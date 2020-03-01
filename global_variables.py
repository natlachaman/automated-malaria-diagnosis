import os


def init():
        
    global NUM_CLASSES
    # including negative class
    NUM_CLASSES = 3
    
    global INPUT_SHAPE
    # INPUT_SHAPE = (256, 256, 3)
    INPUT_SHAPE = (512, 512, 3)
    # INPUT_SHAPE = (768, 768, 3)

    global CLASSES
    CLASSES = {'background': 0,
               'falciparum': 1,
               'white_blood_cell': 2}
    global COLORS
    COLORS = {'pred': (0., 0., 1.),
              'true': (1., 0., 0.)}

    global HOME_FOLDER
    HOME_FOLDER = os.path.join('/', 'home-link', 'knaxq01')

    global DATA_PATH
    DATA_PATH = os.path.join(HOME_FOLDER, 'data', 'quinn')
    
    global IMAGES_PATH
    IMAGES_PATH = os.path.join(DATA_PATH, 'images', 'test')
    
    global PICKLE_FOLDER
    PICKLE_FOLDER = os.path.join(DATA_PATH, 'pickle')
    
    global RESULTS_FOLDER
    RESULTS_FOLDER = os.path.join(HOME_FOLDER, 'Deployment', 'ssd', 'results')


