# %%
import multiprocessing as mp

texts = [
'Ethical/Sustainable Shopping: Addressing customer interest in responsible fashion.',

'Urgent Order Cancellation: Handling a time-sensitive cancellation request.',

'Student Discounts: Providing information about special offers.',

'Product Smell Issue: Addressing a concern about a newly received item\'s odor.',

'Loyalty Program Inquiry: Explaining the customer rewards program.'
]
texts = texts*int(1e6)

def test_func(text):
    a = text[::2]
    b = text[1::2]
    a = b[::2]
    b = a[1::2]
    a = b[2::2]
    b = a[3::2]
    return text[::-1], a, b[::-1]

# %%
if __name__ == '__main__':
    result = []
    with mp.Pool(4) as pool:
        for text in pool.imap(test_func, texts, chunksize=int(2**19)):
            result.append(text)
    # for text in texts:
    #     result.append(test_func(text))

