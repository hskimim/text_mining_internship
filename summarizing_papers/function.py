import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

def extract_text_from_pdf(pdf_path):
    '''
    read pdf file into string type
    '''
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()

    if text:
        return text

def extract_text_from_txt(path):
    with open(path,'rt') as f :
        lines = f.readlines()
    # return ','.join(lines).replace('\n','')
    return ','.join(lines)


def find_start_idx(file_) :
    ls = ['References','REFERENCES','reference','Reference','REFERENCE','참고 문헌','참 고 문 헌','참  고  문  헌' , '참고문헌','<참 고 문 헌>','Reference','reference','REFERENCE']
    start_idx = []
    start_idx = [re.search(str(i),file_).start() for i in ls if re.search(i,file_)]
    if (start_idx != []) and (start_idx[0] < len(file_)) : return start_idx[0]
    else :
        start_idx = [re.search(str(i),file_).start() for i in ls if re.search(i,file_)]
        if (start_idx != []) and (start_idx < len(file_)) : return start_idx[0]
        else :
            print("file didn't find the references.")
            return len(file_)