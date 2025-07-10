import xml.etree.ElementTree as ET
import numpy as np
import cv2

def parse_xml_to_mask(xml_path, image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f" '{image_path}' error")
    
    image_shape = img.shape[:2] 

    tree = ET.parse(xml_path)
    root = tree.getroot()

    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

    for obj in root.findall('object'):
        name = obj.find('name')
        if name is None or name.text.lower() != 'tip burn':
            continue  

        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue  

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, thickness=-1)

    return mask

xml_path = 'E:/U_Net_lettuce_tip_burn/image23.xml'
image_path = 'E:/U_Net_lettuce_tip_burn/image23.jpg'

mask = parse_xml_to_mask(xml_path, image_path)

cv2.imwrite('E:/tip_burn_mask/image23.jpg', mask)


