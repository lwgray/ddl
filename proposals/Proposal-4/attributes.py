import numpy as np

class Attributes(object):
    def __init__(self, product,
                 attributes, dimensions):
        self.attributes = attributes
        self.product = product
        self.dimensions = dimensions
        
    def get_model(self):
        try:
            model = self.attributes.Model
        except AttributeError:
            return np.NaN
        return model

    def get_brand(self):
        try:
            brand = self.attributes.Brand
        except:
            return np.NaN
        return brand

    def get_dimensions(self):
        try:
            height = self.dimensions.Height
            width = self.dimensions.Width
            length = self.dimensions.Length
        except AttributeError:
            return [0,0,0]
        return length, width, height

    def get_length(self):
        try:
            return float(self.dimensions.Length.Value)
        except:
            return 0
        
    def get_width(self):
        try:
            return float(self.dimensions.Width.Value)
        except:
            return 0
        
    def get_height(self):
        try:
            return float(self.dimensions.Height.Value)
        except:
            return 0
    
    def get_salesrank(self):
        try:
            salesrank = int(self.product.SalesRankings.SalesRank[0].Rank)
        except AttributeError:
            return np.NaN
        except IndexError:
            return np.NaN
        return salesrank

    def get_manufacturer(self):
        try:
            manufacturer = self.attributes.Manufacturer
        except AttributeError:
            return np.NaN
        return manufacturer

    def get_max_age(self):
        try:
            max_age = float(self.attributes.ManufacturerMaximumAge.Value)
        except AttributeError:
            max_age = 0
        return max_age

    def get_min_age(self):
        try:
            min_age = float(self.attributes.ManufacturerMinimumAge.Value)
        except AttributeError:
            return 0
        return min_age

    def get_product_group(self):
        try:
            product_group = self.attributes.ProductGroup
        except AttributeError:
            return np.NaN
        return product_group