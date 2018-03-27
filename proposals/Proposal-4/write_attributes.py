from boto.mws.connection import MWSConnection
import numpy as np
import pandas as pd
from itertools import zip_longest
import time
from attributes import Attributes
from boto.exception import BotoServerError
from multiprocessing import Pool
import multiprocessing as mp


# AWS Credentials
aws_access_key_id = 'AKIAJ5D2V2OWY2DZTI3Q'
aws_secret_access_key = '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'
sellerid = 'A11P5SV6O6WK0G'
marketplaceid = 'ATVPDKIKX0DER'
mws = MWSConnection(aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    Merchant=sellerid)


def get_attributes(asin_list):
    '''
    Retrieve certain attributes for all active ASINS from goodstuff database

    Args:
       asin_list(list of lists):
       Each internal list contains 10 ASIN.  This because the mws function
       allows to submit an api request for 10 ASIN at a time.

    Returns:
        master(list): A list of attributes for each ASIN.
        The attributes are:

        identifier: ASIN number
        brand: The manufacturers brand
        pkg_length: The lenght of the package containing the item
        pkg_width: The width of the item.
        pkg_height: The height of the item:
        salesrank:  The selling rate.  lower the numbers the sale fater.
        manufacturer: The company that makes the item
        max_age: The suggested maximum age the item is for (months)
        min_age: The minimum age the item is for (months)
        product_group: Theretail category the item belongs.
    '''

    master = []
    for asins in asin_list:
        try:
            response = mws.get_matching_product(MarketplaceId=marketplaceid,
                                                ASINList=asins)
        except AttributeError:
            print('AttributeError')
            time.sleep(10)
            response = mws.get_matching_product(MarketplaceId=marketplaceid,
                                                ASINList=asins)
        except BotoServerError:
            print('BotoServerError')
            time.sleep(10)
            response = mws.get_matching_product(MarketplaceId=marketplaceid,
                                                ASINList=asins)
        except:
            print('Exception')
            raise

        for result in response._result:
            product = result.Product
            identifier = result['ASIN']
            try:
                attributes = product.AttributeSets.ItemAttributes[0]
            except AttributeError:
                continue
            dimensions = attributes.PackageDimensions
            a = Attributes(product, attributes, dimensions)
            master.append([identifier, a.get_brand(), a.get_length(),
                           a.get_width(), a.get_height(), a.get_salesrank(),
                           a.get_manufacturer(), a.get_max_age(),
                           a.get_min_age(), a.get_product_group()])
    return master


if __name__ == '__main__':
    # import goodstuff data from postgresql database
    data = pd.read_csv('goodstuff.csv.xz')

    # Create list of list containing 10 ASINS per list
    asin_list = list(zip_longest(*[iter(data.asin.unique())]*10))
    # asin_100 = asin_list[:100]
    # asin_10 = asin_list[:10]

    # Perform tasks with multiprocessing
    pool = Pool(processes=8)
    # result = []
    result = pool.map(get_attributes, [asin_list])
    pool.close()
    pool.join()

    # Transform into pandas DataFrame
    attr = pd.DataFrame(result[0], columns=['id', 'brand', 'length', 'width',
                                            'height', 'salesrank',
                                            'manufacturer', 'max_age',
                                            'min_age', 'product_group'])
    # Convert to csv
    attr.to_csv('attributes.csv', header=False)
