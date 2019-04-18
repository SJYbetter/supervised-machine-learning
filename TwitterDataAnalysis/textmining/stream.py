import tweepy
import logging
import time
import os.path
import csv
import pickle

import optparse

LOG = logging.getLogger(__name__)


class RollBook:
    def __init__(self, out_dir, file_name_template):
        # file_name = os.path.splitext(os.path.basename(full_file_name))[0]
        # out_dir = os.path.dirname(file_name)
        self._path = os.path.realpath(out_dir)
        self._base_file_name = file_name_template
        self._file_index = 0
        self._row_count = 0
        self._fp = None
        self._csv_writer = None

        if not os.path.exists(self._path):
            os.mkdir(self._path)
        LOG.info('archive path: %s', self._path)
        self._rollup()

    def _rollup(self):
        if self._fp is not None:
            self._fp.close()
        file_name = os.path.join(self._path, '%s.%d.%s.csv' % (
            self._base_file_name, self._file_index, time.strftime('%y%m%d.%H%M%S')))
        self._fp = open(file_name, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._fp, quoting=csv.QUOTE_ALL, delimiter=';')
        self._file_index += 1
        self._row_count = 0

    def write_row(self, items):
        self._csv_writer.writerow(items)
        self._row_count += 1
        if self._row_count > 20000 or self._fp.tell() > 209715200:
            # roll on file size > 200m or row > 20000
            self._rollup()


class BookList:
    def __init__(self, out_dir):
        self._writers = {}
        self._out_dir = out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)
        self._fp = open(os.path.join(out_dir, 'raw.%s.bin' % (time.strftime('%y%m%d.%H%M%S'),)), 'wb')

    def write(self, file_key, *items):
        if len(items) == 0:
            return
        if file_key not in self._writers:
            writer = self._writers[file_key] = RollBook(self._out_dir, file_key.replace(' ', '_'))
        else:
            writer = self._writers[file_key]
        writer.write_row(items)

    def write_raw(self, status):
        pickle.dump(status, self._fp)


class StatusCollector(tweepy.StreamListener):
    def __init__(self, options, api=None):
        super(StatusCollector, self).__init__(api=api)
        self._opts = options
        self._book = BookList(options.output)

    def collect(self):
        running = True
        streamer = tweepy.Stream(self.api.auth, listener=self, retry_count=100, retry_time=1, buffer_size=16000)
        # while running:
        try:
            # streamer.filter(locations="-114.813613,31.332502,-109.045223,37.003875,-94.617919,33.004106,-89.686924,36.499496,-124.409591,32.534156,-114.131489,42.009518,-109.060062,36.992426,-102.041574,41.003444,-73.727775,40.985171,-71.789356,42.049638,-77.1199,38.791513,-76.909395,38.99511,-85.605165,30.358035,-80.840549,35.000771,-160.555771,18.917466,-154.809379,22.23317,-91.512974,36.970298,-87.495211,42.508302,-88.071449,37.776843,-84.785111,41.760592,-94.043147,28.925011,-88.817017,33.019372,-97.239155,43.499356,-89.489226,49.384358,-91.636942,30.173943,-88.097888,34.996052,-116.049415,44.371038,-104.040114,49.001076,-109.050044,31.332301,-103.001964,37.000104,-104.0489,45.935054,-96.554835,49.000687,-103.002565,33.615765,-94.431215,37.002206,-80.51979,39.719998,-74.689767,42.26986")
            # streamer.filter(track=['python', 'dog', 'car', 'food', 'trump'])
            if self._opts.simple:
                streamer.sample(**build_filter_kwargs(self._opts))
            elif self._opts.filter:
                streamer.filter(**build_filter_kwargs(self._opts))
            else:
                LOG.warning("please use --simple or --filter option")
                # break
        except KeyboardInterrupt:
            running = False
        except Exception as e:
            running = False
            LOG.exception("Unexpected exception.", exc_info=e)
            # if running:
            #     LOG.debug('Sleeping...')
            #     time.sleep(5)

    def on_connect(self):
        LOG.info("connected now")

    def _persist_tweet(self, status):
        try:
            tid = status.id
            text = status.text
            tweet_place = status.place.full_name if status.place is not None else ''
            user_id = status.user.id if status.user is not None else ''
            location = status.user.location if status.user.location is not None else ''
            # timestamp = time.strptime(status.created_at, '%a %b %d %H:%M:%S %z %Y')
            timestamp = status.created_at.strftime('%y-%m-%d %H:%M:%S')
            self._book.write('tweets', tid, timestamp, user_id, tweet_place.strip(), location.strip(),
                             text.strip().replace('\n', ' '))
        except Exception as e:
            LOG.exception('persist tweet error', exc_info=e)

    def on_status(self, status):
        if hasattr(status, 'retweeted_status'):
            self._persist_tweet(status.retweeted_status)
        self._persist_tweet(status)

        # self._book.write('unclasification', status.id, status.text)
        self._book.write_raw(status)

    def on_error(self, status_code):
        LOG.error('listener error: %d', status_code)

    def on_disconnect(self, stream_data):
        LOG.warning('disconnect: %s', stream_data)
        # msg = json.loads(stream_data)
        # logger.warn("Disconnect: code: %d stream_name: %s reason: %s",
        # utils.resolve_with_default(msg, 'disconnect.code', 0),
        # utils.resolve_with_default(msg, 'disconnect.stream_name', 'n/a'),
        # utils.resolve_with_default(msg, 'disconnect.reason', 'n/a'))


def build_filter_kwargs(options):
    result = {'stall_warnings': True}
    if options.lang is not None and len(options.lang) > 0:
        result['languages'] = options.lang
    if options.locations is not None and len(options.locations) > 0:
        from geo import MemoryDb
        locations = []
        db = MemoryDb()
        for name in options.locations:
            place = db.find(name)
            if place is None:
                LOG.warning('location %s not found', name)
            else:
                locations.extend(place.bounding_box.origin())
                locations.extend(place.bounding_box.corner())
        if len(locations) > 0:
            result['locations'] = locations
    if options.track is not None and len(options.track) > 0:
        result['track'] = options.track
    return result


def create_option(parser):
    def foo_callback(option, opt_str, value, parser, sep=','):
        if value is None or value == opt_str:
            if not hasattr(parser.values, option.dest):
                setattr(parser.values, option.dest, None)
            return
        values = list(map(str.strip, value.split(sep)))
        prop = getattr(parser.values, option.dest)
        if prop is None:
            setattr(parser.values, option.dest, values)
        else:
            prop.extend(values)

    group = optparse.OptionGroup(parser, 'streaming flags')
    group.add_option('-s', '--simple', action='store_true', default=False,
                     help='sample realtime Tweets, see https://developer.twitter.com/en/docs/tweets/sample-realtime'
                          '/api-reference/get-statuses-sample')
    group.add_option('-f', '--filter', action='store_true', default=False,
                     help="filter realtime Tweets, see https://developer.twitter.com/en/docs/tweets/filter-realtime"
                          "/api-reference/post-statuses-filter")

    group.add_option('--lang', type='string', action='callback', nargs=1, metavar='en,jp...',
                     callback=foo_callback, help="bcp47 language code list")

    group.add_option('--locations', type='string', action='callback', nargs=1, metavar='"North Dakota,USA; Kansas,USA"',
                     callback=foo_callback, callback_kwargs={'sep': ';'},
                     help='location name, etc: Virgin Islands, USA, New York, Delaware, USA')

    group.add_option('--track', type='string', action='callback', nargs=1, metavar='trump, facebook, nlp',
                     callback=foo_callback, help='topic keyword for tracing')

    group.add_option('-o', '--output', metavar='directroy', default='dataset',
                     help="directory for realtime tweet write to")
    return group
