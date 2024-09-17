import xml.etree.ElementTree as ET
import re
import pandas as pd

from tqdm import tqdm
# https://docs.python.org/3/library/xml.etree.elementtree.html

tree = ET.parse('ecatalog-service_nmarket_pro.xml')
root = tree.getroot()

output = {}
def _bredthwiseXML(childs):
    output = dict()
    for sbch in childs:
        second_key = sbch.tag
        second_key = re.compile(r"\{.+?\}").sub("", second_key)
        if len(sbch) == 0:
            output.update({second_key: sbch.text})
        else:
            output.update({second_key: [i for i in sbch]})
    return output

def fullXML(childs, show=1):
    output = _bredthwiseXML(childs)
    while True:
        to_add = dict()
        for key in output:
            if isinstance(output[key], list):
                break_ = False
                new_vals = _bredthwiseXML(output[key])
                for subkey in new_vals:
                    to_add.update({f"{key}_{subkey}": new_vals[subkey]})
                output[key] = []
        if show:
            print(to_add)
            print("*"*40)
        for newkey in to_add:
            output.update({newkey: to_add[newkey]})
        if len(to_add) == 0:
            break
    output = {i: output[i] for i in output if output[i]}
    return output

fin_output = dict()
for child in tqdm(root, position=0, leave=True, desc='Parse XML'):
    if 'internal-id' in child.attrib:
        fin_output.update({child.attrib['internal-id']: fullXML(child, show=0)})

fin_output_df = pd.DataFrame(fin_output).transpose()
fin_output_df.to_excel(r"full_result_2024_08_18.xlsx")
print(fin_output_df.shape)