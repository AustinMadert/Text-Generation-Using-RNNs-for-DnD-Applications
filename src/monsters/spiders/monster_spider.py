import scrapy
from ..items import MonsterItem
import time
from functools import reduce


query = 'bestiary/'


class MonsterSpider(scrapy.Spider):
    name = 'monster_spider'
    allowed_domains = ['chisaipete.github.io']
    start_urls = ['http://chisaipete.github.io/' + query]


    def parse(self, response):

        names = response.xpath('//a[@class="post-link"]/text()').getall()
        links = [response.url + 'creature/' + str(i).lower() for i in names]
        for url in links:
            yield scrapy.Request(url, callback=self.parse_statblock)

    
    def parse_statblock(self, response):
        item = MonsterItem()

        item['name'] = response.xpath('//div[@class="creature-heading"]/h1/text()').get()
        item['short_desc'] = response.xpath('//div[@class="creature-heading"]/h2/text()').get()
        item['armor_class'] = response.xpath('//div[@class="property-line first"]/p/text()').get()
        item['speed'] = response.xpath('//div[@class="property-line last"]/p/text()').get()
        item['strength'] = response.xpath('//div[@class="ability-strength"]/p/text()').get()
        item['dexterity'] = response.xpath('//div[@class="ability-dexterity"]/p/text()').get()
        item['constitution'] = response.xpath('//div[@class="ability-constitution"]/p/text()').get()
        item['intelligence'] = response.xpath('//div[@class="ability-intelligence"]/p/text()').get()
        item['wisdom'] = response.xpath('//div[@class="ability-wisdom"]/p/text()').get()
        item['charisma'] = response.xpath('//div[@class="ability-charisma"]/p/text()').get()
        item['actions'] = reduce(lambda x, y: x + '|' + y, response.xpath('//div[@class="section-left"]/p/text()').getall())
        item['url'] = response.url

        skills = response.xpath('//div[@class="property-line"]/p/text()').getall()
        item['hit_points'] = skills[0]
        item['skills'] = skills[1]
        item['languages'] = skills[2]
        item['challenge'] = skills[3]

        yield item

