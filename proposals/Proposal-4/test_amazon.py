# [Credentials]
aws_access_key_id = 'AKIAJ5D2V2OWY2DZTI3Q'
aws_secret_access_key = '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'
sellerid = 'A11P5SV6O6WK0G'
marketplaceid = 'ATVPDKIKX0DER'

from boto.mws.connection import MWSConnection
import pandas as pd
from amazonproduct import API
from collections import OrderedDict
from decimal import Decimal
from test_calculator import calculate
from test_sortit import sortit
import argparse
from test_categories import categories
from decimal import InvalidOperation
from test_signature import amazon_test_url
import xmltodict as xd
import requests
import time


def get_products(search_index, keywords, pages):
    api = API(locale='us')
    mws = MWSConnection(Merchant=sellerid)
    final = []
    for page in range(1, (pages + 1)):
        try:
            productlist = api.item_search(search_index, Keywords=keywords, Sort='salesrank', paginate=False, ItemPage=page, ResponseGroup='Offers', MerchandId='Amazon')
        except:
            continue
        products = [items for items in productlist.Items.Item]
        solid = []
        columns = ['ASIN', 'AMAZON', 'FBA', 'MERCHANT']
        for i, product in enumerate(products):
            try:
                a = product.ASIN.text
                b = product.Offers.Offer.OfferListing.Price.FormattedPrice.text.lstrip('$')
                solid.append([a, b, 0, 0])
            except:
                continue

        data = pd.DataFrame(solid, columns=columns)

        for asin in data['ASIN']:
            index = data.index[data['ASIN'] == asin]
            try:
                amazon_price = float(data['AMAZON'][index].values[0])
            except ValueError:
                try:
                    amazon_price = float(data['AMAZON'][index])
                except ValueError:
                    continue
            # print amazon_price

            # asin =  data.ASIN.values.item()

            # get hmac256 signed url for api request
            x = amazon_test_url(asin)[0]
            r = requests.get(x)
            d = xd.parse(r.content)

            print asin
            try:
                prices = [(x['IsFulfilledByAmazon'], (Decimal(x['ListingPrice']['Amount']) + Decimal(x['Shipping']['Amount']))) for x in d['GetLowestPricedOffersForASINResponse']['GetLowestPricedOffersForASINResult']['Offers']['Offer']]
            except TypeError:
                try:
                    z = d['GetLowestPricedOffersForASINResponse']['GetLowestPricedOffersForASINResult']['Offers']['Offer']
                    prices = [(z['IsFulfilledByAmazon'], (Decimal(z['ListingPrice']['Amount']) + Decimal(z['Shipping']['Amount'])))]
                except TypeError:
                    continue
            except KeyError:
                try:
                    print "Sleeping 1"
                    time.sleep(1830)
                    z = d['GetLowestPricedOffersForASINResponse']['GetLowestPricedOffersForASINResult']['Offers']['Offer']
                    prices = [(z['IsFulfilledByAmazon'], (Decimal(z['ListingPrice']['Amount']) + Decimal(z['Shipping']['Amount'])))]
                except KeyError:
                    print "Sleeping 2"
                    time.sleep(1830)
                    z = d['GetLowestPricedOffersForASINResponse']['GetLowestPricedOffersForASINResult']['Offers']['Offer']
                    prices = [(z['IsFulfilledByAmazon'], (Decimal(z['ListingPrice']['Amount']) + Decimal(z['Shipping']['Amount'])))]

            try:
                merchant_price = [x[1] for x in prices if x[0] == 'false'][0]
            except IndexError:
                merchant_price = 0

            data.at[index, 'MERCHANT'] = merchant_price

            try:
                fba_price = [Decimal(x[1]).quantize(Decimal('.01')) for x in prices if x[0] == 'true']
            except:
                fba_price = []

            if fba_price:
                try:
                    if len(fba_price) == 1:
                        if fba_price[0] == amazon_price:
                            fba_price = merchant_price
                    else:
                        fba_price.remove(Decimal(amazon_price).quantize(Decimal('.01')))
                        fba_price = fba_price[0]
                except ValueError:
                    fba_price = fba_price[0]
                except IndexError:
                    if merchant_price != 0:
                        fba_price = merchant_price
                    else:
                        fba_price = 0
            elif not fba_price and merchant_price != 0:
                fba_price = merchant_price
            else:
                fba_price = 0

            data.at[index, 'FBA'] = fba_price

            row = get_attributes(data, index.values[0], mws)
            final.append(row)

    columns = ['asin', 'title1', 'title2', 'asin1', 'price', 'fba', 'merchant',
               'salesrank', 'productgroup', 'height', 'length', 'width',
               'weight', 'rf', 'vcf', 'promo', 'image', 'pagelink',
               'storepick', 'shipping']

    final_df = pd.DataFrame(final, columns=columns)
    return final_df


def get_attributes(data, index, mws):
    response = mws.get_matching_product_for_id(MarketplaceId=marketplaceid,
                                               IdType='ASIN',
                                               IdList=[data['ASIN'][index]])
    result = response._result[0]
    product = result.Products.Product[0]
    identifiers = product.Identifiers
    attributes = product.AttributeSets.ItemAttributes[0]
    dimensions = attributes.PackageDimensions

    attr = OrderedDict()
    # These should be functions with decorators for encoding and try/except
    attr['asin'] = identifiers.MarketplaceASIN.ASIN
    attr['title1'] = attributes.Title
    attr['title2'] = attr['title1']
    attr['asin1'] = attr['asin']

    try:
        attr['price'] = Decimal(data['AMAZON'][index]).quantize(Decimal('.01'))
    except InvalidOperation:
        attr['price'] = Decimal(0)
    except TypeError:
        attr_price = data['AMAZON'][index].astype(Decimal)
        print type(attr_price)
        attr['price'] = Decimal(attr_price).quantize(Decimal('.01'))

    attr['fba'] = Decimal(data['FBA'][index])

    try:
        attr['merchant'] = Decimal(data['MERCHANT'][index])
    except TypeError:
        attr_merchant = data['MERCHANT'][index].astype(Decimal)
        attr['merchant'] = Decimal(attr_merchant).quantize(Decimal('.01'))

    try:
        attr['salesrank'] = product.SalesRankings.SalesRank[0].Rank.encode('ascii', errors='ignore')
    except:
        attr['salesrank'] = 0

    attr['productgroup'] = attributes.ProductGroup.encode('ascii',
                                                          errors='ignore')
    try:
        attr['height'] = dimensions.Height.Value
        attr['length'] = dimensions.Length.Value
        attr['width'] = dimensions.Width.Value
    except AttributeError:
        attr['height'] = 0
        attr['length'] = 0
        attr['width'] = 0

    try:
        attr['weight'] = dimensions.Weight.Value
    except AttributeError:
        attr['weight'] = 0

    try:
        category = categories[attr['productgroup']]
    except:
        category = (0.15, 1)

    attr['rf'] = category[0]
    attr['vcf'] = category[1]
    attr['promo'] = ''
    attr['image'] = attributes.SmallImage.URL
    attr['pagelink'] = None
    attr['storepick'] = 'no'
    attr['shipping'] = 'yes'

    final = attr.values()
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find Amazon Products')
    parser.add_argument('--index', default='Toys',
                        help='Product Category')
    parser.add_argument('--keywords', default='-kjdflj',
                        help='Search terms')
    parser.add_argument('--pages', type=int, default=1,
                        help='The num of pages to be return;10 items per page')
    parser.add_argument('--output', default='final.csv',
                        help='The file that data should be writeen to')
    args = parser.parse_args()
    products = get_products(args.index, args.keywords, args.pages)
    numbers = calculate(products)
    s = sortit(numbers, args.output)
    print(s)
