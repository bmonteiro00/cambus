import argparse

class ArgParserMqtt:

    def __init__(self,caPath, certPath, keyPath, topic, subscribeTo, jsonString, clientId):

        self._parser = None
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument("-r", "--rootCA", action="store", required=False, dest="rootCAPath",
                                  default=caPath,
                                  help="Root CA file path")
        self._parser.add_argument("-c", "--cert", action="store", dest="certificatePath",
                                  default=certPath,
                                  help="Certificate file path")
        self._parser.add_argument("-k", "--key", action="store", dest="privateKeyPath",
                                  default=keyPath,
                                  help="Private key file path")
        self._parser.add_argument("-id", "--clientId", action="store", dest="clientId", default=clientId,
                                  help="Targeted client id")
        self._parser.add_argument("-t", "--topic", action="store", dest="topic", default=topic,
                                  help="Targeted topic")

    def getArgParser(self):
        return self._parser