import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from scipy.ndimage.measurements import center_of_mass
import cv2
import numpy as np
import os
from tifffile import imsave
from scipy.sparse import save_npz, load_npz, hstack
import gc
import h5py
import nrrd


class CaImAn:

    def __init__(self, path, fileName):
        self.initializeThreads()
        self.c, self.dview, self.n_processes = None, None, None
        self.path, self.fileName = path, fileName
        self.fullPath = [path + fileName]
        self.datasetParams = {'fr': 1.93, 'decay_time': 3.5}
        self.rigidMotionCorrectionParams = {'max_shifts': (40, 40), 'strides': (48, 48), 'overlaps': (24, 24),
                                            'num_frames_split': 30, 'max_deviation_rigid': 4, 'pw_rigid': False}
        self.piecewiseMotionCorrectionParams = {'max_shifts': (40, 40), 'strides': (48, 48), 'overlaps': (24, 24),
                                                 'num_frames_split': 30, 'max_deviation_rigid': 4, 'pw_rigid': True,
                                                 'shifts_opencv': True, 'border_nan': 'copy'}
        self.segmentationParams = {'p': 2, 'nb': 2, 'merge_thr': 0.85, 'rf': 20, 'stride': 3, 'K': 20,
                                   'gSig': [3, 3], 'method_init': 'greedy_roi', 'ssub': 1, 'tsub': 1, 'min_SNR': 0,
                                   'SNR_lowest': 0, 'rval_thr': 0.85, 'rval_lowest': 0.1, 'min_cnn_thr': 0.99,
                                   'cnn_lowest': 0.1}
        self.otherParams = {'rolling_sum': True, 'only_init': True, 'use_cnn': True}
        self.parametersObject = None
        self.labels, self.segmented, self.centroids = None, None, None
        self.averageFrame = None

    def initializeThreads(self):
        try:
            cv2.setNumThreads(0)
        except:
            pass

    def startCluster(self):
        if self.dview is not None:
            self.stopCluster()
        self.c, self.dview, self.n_processes = cm.cluster.setup_cluster(
            backend='local', n_processes=None, single_thread=False)

    def stopCluster(self):
        cm.stop_server(dview=self.dview)
        gc.collect()

    def setupParameters(self, rigid):
        if rigid:
            paramsList = [self.datasetParams, self.rigidMotionCorrectionParams, self.segmentationParams,
                          self.otherParams]
        else:
            paramsList = [self.datasetParams, self.piecewiseMotionCorrectionParams, self.segmentationParams,
                          self.otherParams]
        paramsDict = {}
        for parameters in paramsList:
            for key in parameters.keys():
                paramsDict[key] = parameters[key]
        self.parametersObject = params.CNMFParams(params_dict=paramsDict)

    def correctMotion(self, rigid=False):
        self.startCluster()
        if rigid is False:
            self.setupParameters(rigid=False)
        else:
            self.setupParameters(rigid=True)
        self.mc = MotionCorrect(self.fullPath, dview=self.dview, **self.parametersObject.get_group('motion'))
        self.mc.motion_correct(save_movie=True)
        self.border_to_0 = 0 if self.mc.border_nan is 'copy' else self.mc.border_to_0

    def computeSegmentation(self):
        fname_new = cm.save_memmap(self.mc.mmap_file, base_name='memmap_', order='C',
                                   border_to_0=self.border_to_0, dview=self.dview)
        Yr, self.dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(self.dims), order='F')
        self.averageFrame = np.mean(images, axis=0)
        self.startCluster()
        cnm = cnmf.CNMF(self.n_processes, params=self.parametersObject, dview=self.dview)
        cnm = cnm.fit(images)
        self.cnm2 = cnm.refit(images, dview=self.dview)
        self.cnm2.estimates.evaluate_components(images, self.cnm2.params, dview=self.dview)
        self.cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
        self.cnm2.estimates.select_components(use_object=True)

        self.computeSegmentationImages(self.cnm2.estimates.A, self.dims)
        self.computeNeuronCentroids(self.cnm2.estimates.A, self.dims)

    def computeSegmentationImages(self, components, dims):
        segmented, labels = np.zeros(dims), np.zeros(dims)
        for i in range(components.shape[1]):
            component = np.reshape(components[:, i].toarray(), dims, order='F')
            segmented += component
            labels[component > 0] = i + 1
        self.segmented, self.labels = self.convertTo16bit(segmented), labels

    def computeNeuronCentroids(self, components, dims):
        centroids = np.zeros((components.shape[1], 2))
        for i in range(components.shape[1]):
            component = np.reshape(components[:, i].toarray(), dims, order='F')
            centroids[i, :] = center_of_mass(component)
        self.centroids = centroids

    def deleteMemoryMappedFiles(self):
        for file in os.listdir(self.path):
            if '.mmap' in file:
                os.remove(self.path + '/' + file)

    def saveFilm(self, fileName):
        correctedMovie = cm.load(self.mc.mmap_file)
        convertedMovie = self.convertTo16bit(correctedMovie)
        imsave(self.path + fileName, convertedMovie)

    def saveResults(self, tag='', components=True, image=True, default=False):
        if default:
            self.cnm2.save(self.path + 'results' + tag + '.hdf5')
        else:
            file = h5py.File(self.path + 'results' + tag +'.hdf5', 'w')
            file.create_dataset('labels', data=self.labels)
            file.create_dataset('centroids', data=self.centroids)
            file.create_dataset('timeSeries', data=self.cnm2.estimates.F_dff)
            file.create_dataset('deconvolved', data=self.cnm2.estimates.C)
            file.create_dataset('spikes', data=self.cnm2.estimates.S)
            file.create_dataset('SNR', data=self.cnm2.estimates.SNR_comp)
            file.create_dataset('dims', data=self.dims)
            file.create_dataset('averageFrame', data=self.averageFrame)
            file.close()
        if components:
            save_npz(self.path + 'components' + tag + '.npz', self.cnm2.estimates.A)
        if image:
            imsave(self.path + 'segmentation' + tag + '.tif', self.segmented)

    @staticmethod
    def convertTo16bit(array):
        if np.amin(array >= 0):
            array -= np.amin(array)
        else:
            array[array < 0] = 0
        array *= 65535 / np.amax(array)
        return array.astype('uint16')

    @staticmethod
    def loadResults(path):
        return CaimanResults(path)


class CaimanResults:

    def __init__(self, path):
        file = h5py.File(path, 'r')
        self.timeSeries = np.array(file['timeSeries'])
        self.deconvolved = np.array(file['deconvolved'])
        self.spikes = np.array(file['spikes'])
        self.SNR = np.array(file['SNR'])
        self.centroids = np.array(file['centroids'])
        self.labels = np.array(file['labels'])
        self.dims = np.array(file['dims'])
        self.averageFrame = np.array(file['averageFrame'])
        file.close()


class Compiler:

    def __init__(self, path, keywords):
        self.path = path
        self.keywords = keywords
        self.results = None
        self.stack = None
        self.components = None

    def loadAllPlanes(self):
        files = os.listdir(self.path)
        results = {}
        for file in files:
            nKeywords = self.countKeywordsInFileName(file, self.keywords)
            if ('plane' in file) and ('.hdf5' in file):
                if nKeywords == len(self.keywords):
                    chunk = file.split('plane')[1]
                    nDigits = 0
                    for i in range(len(chunk)):
                        try:
                            int(chunk[i])
                            nDigits += 1
                        except:
                            break
                    plane = chunk[:nDigits]
                    results[int(plane)] = CaImAn.loadResults(self.path + file)
        self.results = results

    def loadAllComponents(self):
        files = os.listdir(self.path)
        components = {}
        for file in files:
            nKeywords = self.countKeywordsInFileName(file, self.keywords)
            if ('components' in file) and ('.npz' in file):
                if nKeywords == len(self.keywords):
                    chunk = file.split('plane')[1]
                    nDigits = 0
                    for i in range(len(chunk)):
                        try:
                            int(chunk[i])
                            nDigits += 1
                        except:
                            break
                    plane = chunk[:nDigits]
                    components[int(plane)] = load_npz(self.path + file)
        self.components = components

    def compileAndSave(self):
        results = self.results
        planes = np.sort(list(results.keys()))
        plane = planes[0]
        timeSeries = results[plane].timeSeries
        planeVector = np.array([plane] * timeSeries.shape[0])
        deconvolved = results[plane].deconvolved
        spikes = results[plane].spikes
        SNR = results[plane].SNR
        centroids = results[plane].centroids
        labels = np.expand_dims(results[plane].labels, axis=2)
        self.stack = np.expand_dims(results[plane].averageFrame, axis=2)
        for plane in planes[1:]:
            timeSeries = np.append(timeSeries, results[plane].timeSeries, axis=0)
            planeVector = np.append(planeVector, [plane] * results[plane].timeSeries.shape[0])
            deconvolved = np.append(deconvolved, results[plane].deconvolved, axis=0)
            spikes = np.append(spikes, results[plane].spikes, axis=0)
            SNR = np.append(SNR, results[plane].SNR, axis=0)
            centroids = np.append(centroids, results[plane].centroids, axis=0)
            labels = np.append(labels, np.expand_dims(results[plane].labels, axis=2), axis=2)
            self.stack = np.append(self.stack, np.expand_dims(results[plane].averageFrame, axis=2), axis=2)
        fileName = 'results'
        for keyword in self.keywords:
            fileName += '_' + keyword
        fileName += '.hdf5'
        file = h5py.File(self.path + fileName, 'w')
        file.create_dataset('labels', data=labels)
        file.create_dataset('centroids', data=centroids)
        file.create_dataset('timeSeries', data=timeSeries)
        file.create_dataset('deconvolved', data=deconvolved)
        file.create_dataset('spikes', data=spikes)
        file.create_dataset('SNR', data=SNR)
        file.create_dataset('planes', data=planeVector)
        file.create_dataset('averageFrames', data=self.stack)
        file.close()

    def compileAndSaveComponents(self):
        planes = np.sort(list(self.components.keys()))
        C = self.components[planes[0]]
        for plane in planes[1:]:
            C = hstack([C, self.components[plane]])
        fileName = 'components'
        for keyword in self.keywords:
            fileName += '_' + keyword
        fileName += '.npz'
        save_npz(self.path + fileName, C)

    def saveFilmStack(self):
        fileName = 'filmStack'
        for keyword in self.keywords:
            fileName += '_' + keyword
        fileName += '.nrrd'
        nrrd.write(self.path + fileName, np.transpose(self.stack.astype('uint16'), (1, 0, 2)))

    @staticmethod
    def countKeywordsInFileName(fileName, keywords):
        nKeywords = 0
        for keyword in keywords:
            if keyword in fileName:
                nKeywords += 1
        return nKeywords


def normalize(vector):
    vector = np.array(vector)
    vector -= min(vector)
    vector /= max(vector)
    return vector


if __name__ == '__main__':

    path = '/home/tonepone/Documents/DataPDK/'
    subjects = ['fish1']
    experiments = ['df1']
    planes = [1, 2, 3, 4, 5, 6, 7, 8]
    tag = '_{}_{}_plane{}'

    for fish in subjects:
        for experiment in experiments:
            for plane in planes:
                caiman = CaImAn(path, '{}_{}.raw #{}.tif'.format(fish, experiment, plane))
                caiman.correctMotion()
                caiman.saveFilm('corrected' + tag.format(fish, experiment, plane) + '.tif')
                os.remove(path + '{}_{}.raw #{}.tif'.format(fish, experiment, plane))
                caiman.computeSegmentation()
                caiman.saveResults(tag=tag.format(fish, experiment, plane))
                caiman.deleteMemoryMappedFiles()
                caiman.stopCluster()

    compile = False
    if compile:
        compiler = Compiler(path, keywords=['fish1', 'df1'])
        compiler.loadAllPlanes()
        compiler.compileAndSave()
        compiler.saveFilmStack()
        compiler.loadAllComponents()
        compiler.compileAndSaveComponents()


