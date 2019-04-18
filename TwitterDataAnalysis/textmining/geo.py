import logging
import os.path
import pickle


class LocalDb:
    """ cache location info in memory """

    def __init__(self):
        self._places = {}

    def put(self, place):
        if place is None:
            raise RuntimeError("place is None")

        self._cache(place.id, place)
        self._persist_put(place.id, place)

    def get(self, place_id):
        if place_id in self._places:
            return self._places[place_id]
        return self._cache(place_id, self._persist_get(place_id))

    def find(self, name):
        # find in memory cache
        for place in self._places.values():
            if place.full_name == name or place.name == name:
                return place
        # find in persist storage
        place = self._persist_find(name)
        if place is None:
            return None
        # cache it
        return self._cache(place.id, place)

    def _cache(self, place_id, place):
        if place is not None:
            self._places[place_id] = place
        return place

    def _persist_put(self, place_id, place):
        pass

    def _persist_get(self, place_id):
        pass

    def _persist_find(self, name):
        pass

    def __getitem__(self, place_id):
        return self.get(place_id)

    def __iter__(self):
        return self._places.__iter__()


class MongoDb(LocalDb):
    """ persist location into mongodb """
    logger = logging.getLogger('mongodb_db')

    def __init__(self):
        super(MongoDb, self).__init__()
        import pymongo as mgo
        self._mongodb = mgo.MongoClient()
        self._tab = self._mongodb['nlp'].locations

    def __del__(self):
        self._mongodb.close()

    def _persist_get(self, place_id):
        place = self._tab.find_one({'_id': place_id})
        return MongoDb._place_from_db(place)

    def _persist_put(self, place_id, place):
        serial_place = MongoDb._place_to_dict(place)
        self._tab.find_one_and_replace({'_id': place.id}, serial_place, upsert=True)
        MongoDb.logger.info('persist place: %s', place.full_name)

    def _persist_find(self, name):
        place = self._tab.find_one({'name': name})
        return MongoDb._place_from_db(place)

    @staticmethod
    def _place_from_db(doc):
        if doc is None:
            return None
        return pickle.loads(doc['binary'])

    @staticmethod
    def _place_to_dict(place):
        from bson.binary import Binary
        result = {'country': place.country,
                  'country_code': place.country_code,
                  'full_name': place.full_name,
                  'id': place.id,
                  '_id': place.id,
                  'name': place.name,
                  'place_type': place.place_type,
                  'url': place.url,
                  'bounding_box': {
                      'type': place.bounding_box.type,
                      'coordinates': place.bounding_box.coordinates
                  },
                  'container': [],
                  'binary': Binary(pickle.dumps(place))
                  }
        if hasattr(place, 'contained_within'):
            for c in place.contained_within:
                result['container'].append(c.id)
        return result


class MemoryDb(LocalDb):
    """ persist location info into file """
    _cache_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'geo.dat'))
    logger = logging.getLogger('memory_db')

    def __init__(self):
        super(MemoryDb, self).__init__()
        self._places = MemoryDb.load() or {}
        self._updated = False

    def __del__(self):
        if self._updated:
            MemoryDb.save(self._places)

    def _persist_put(self, place_id, place):
        self._updated = True

    def reload(self):
        self._places = MemoryDb.load() or {}

    def copy_to(self, newdb):
        for place in self._places.values():
            newdb.put(place)

    @staticmethod
    def load():
        if not os.path.exists(MemoryDb._cache_file):
            return None
        MemoryDb.logger.info("load geo data from file: %s", MemoryDb._cache_file)
        with open(MemoryDb._cache_file, 'rb') as fp:
            return pickle.load(fp)

    @staticmethod
    def save(places):
        MemoryDb.logger.info("save geo data to file: %s", MemoryDb._cache_file)
        with open(MemoryDb._cache_file, 'wb') as fp:
            pickle.dump(places, fp)


class GeoData(object):
    __states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
                "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
                "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
                "Montana", "Nebraska",
                "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
                "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
                "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin",
                "Wyoming"]

    # __states = ["Alabama", "Alaska", "Arizona"]

    def __init__(self, api, db):
        self._api = api
        self._db = db

    def __skip_loaded(self):
        nothing = []
        for name in GeoData.__states:
            place = self._db.find(name)
            if place is None:
                nothing.append(name)
        return nothing

    def load(self):
        absent_states = self.__skip_loaded()
        for i in range(0, len(absent_states), 8):
            places = self._api.geo_search(query=','.join(absent_states[i:i + 8]), granularity='admin',
                                          contained_within='96683cc9126741d1', max_result=500)
            for p in places:
                self._db.put(p)
                if not hasattr(p, 'contained_within'):
                    continue
                for c in p.contained_within:
                    self._db.put(c)


if __name__ == "__main__":
    # from login import api

    # GeoData(api, MemoryDb()).load()

    db = MemoryDb()
    db.copy_to(MongoDb())
