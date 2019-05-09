# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class MonsterItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    name = scrapy.Field()
    short_desc = scrapy.Field()
    armor_class = scrapy.Field()
    hit_points = scrapy.Field()
    speed = scrapy.Field()
    strength = scrapy.Field()
    dexterity = scrapy.Field()
    constitution = scrapy.Field()
    intelligence = scrapy.Field()
    wisdom = scrapy.Field()
    charisma = scrapy.Field()
    skills = scrapy.Field()
    languages = scrapy.Field()
    challenge = scrapy.Field()
    actions = scrapy.Field()
    url = scrapy.Field()