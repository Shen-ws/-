from utils import load_paper_info, filter_papers
import pandas as pd

src_file = 'cvpr2023.csv'

keywords = [
    'person re-identification',
    're-id',
    'person search',
    'reidentification',
    're-identification',
    'identification',
]

reversed_keywords = [
    '3d',
    'car',
    'object'
]

paper_infos = load_paper_info(src_file)
paper_infos = filter_papers(paper_infos, keywords, reversed_keywords)

print('The total number of papers is', len(paper_infos))
if len(paper_infos) > 0:
    df = pd.DataFrame.from_dict(paper_infos)
    df.to_csv(f'filted_cvpr2023.csv', index=True, header=True)