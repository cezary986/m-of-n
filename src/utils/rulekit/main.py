import os
import jpype
import jpype.imports

from typing import List
import glob
import logging
import zipfile
import re
from enum import Enum
from subprocess import Popen,PIPE,STDOUT


class JRE_Type(Enum):
    """:meta private:"""
    open_jdk = 'open_jdk'
    oracle = 'oracle'


class RuleKit:
    """Class used for initializing RuleKit
    
    It starts JVM underhood and setups it with jars. 

    Attributes
    ----------
    version : str
        version of RuleKit jar used by wrapper (not equal to python package version).
    """
    version: str
    initialized: bool = False
    _logger: logging.Logger = None
    _jar_dir_path: str
    _class_path: str
    _rulekit_jar_file_path: str
    _jre_type: JRE_Type

    @staticmethod
    def _detect_jre_type():
        try:
            output = Popen(["java", "-version"],stderr=STDOUT,stdout=PIPE)
            exit_code = output.returncode
            output = str(output.communicate()[0])
            if 'openjdk' in output:
                RuleKit._jre_type = JRE_Type.open_jdk
            else:
                RuleKit._jre_type = JRE_Type.oracle
        except FileNotFoundError as error:
            raise Exception('RuletKit requires java JRE to be installed (version 1.8.0 recommended)')


    @staticmethod
    def init(jar_file_path: str = None, initial_heap_size: int = None, max_heap_size: int = None):
        """Initialize package.

        This method must by called before using any operators in this package.
        It configure and starts JVM and load RuleKit jar file.

        Parameters
        ----------
        jar_file_path : str
            path to RuleKit jar file
        initial_heap_size : int
            JVM initial heap size in mb
        max_heap_size : int
            JVM max heap size in mb

        Raises
        ------
        Exception
            If failed to load RuleKit jar file.
        """
        RuleKit._setup_logger()
        if RuleKit.initialized and jar_file_path:
            if  jar_file_path == RuleKit._rulekit_jar_file_path:
                RuleKit._logger.warn('Tried to call RuleKit.init() multiple times, RuleKit already initialized with the same jar file')
                return
            else:
                RuleKit._logger.info('Initializing Rulekit again with new jar file path')
                jpype.shutdownJVM()

        RuleKit._detect_jre_type()
        current_path: str = os.path.dirname(os.path.realpath(__file__))
        RuleKit._jar_dir_path = f"{current_path}/jar"
        class_path_separator = os.pathsep
        try:
            jars_paths: List[str] = glob.glob(f"{RuleKit._jar_dir_path}/*.jar")
            RuleKit._class_path = f'{str.join(class_path_separator, jars_paths)}'
            if jar_file_path is not None:
                jars_paths = glob.glob(f"{RuleKit._jar_dir_path}/*.jar")
                jars_paths.append(jar_file_path)
                RuleKit._class_path = f'{str.join(class_path_separator, jars_paths)}'
                RuleKit._rulekit_jar_file_path = jar_file_path
            else:
                RuleKit._rulekit_jar_file_path = list(filter(lambda path: 'rulekit' in os.path.basename(path), jars_paths))[0]
            RuleKit._read_versions()
            RuleKit._launch_jvm(initial_heap_size, max_heap_size)
            RuleKit.initialized = True
        except IndexError as error:
            RuleKit._logger.error('Failed to load jar files')
            raise Exception('''\n
Failed to load RuleKit jar file. Check if valid rulekit jar file is present in "rulekit/jar" directory.

If you're running this packae for the first time you need to download RuleKit jar file by running:
    python -m rulekit download_jar
        ''')


    @staticmethod
    def _setup_logger():
        logging.basicConfig()
        RuleKit._logger = logging.getLogger('RuleKit')

    @staticmethod
    def _read_versions():
        jar_archive = zipfile.ZipFile(RuleKit._rulekit_jar_file_path, 'r')
        try:
            manifest_file_content: str = jar_archive.read('META-INF/MANIFEST.MF').decode('utf-8')
            RuleKit.version = re.findall(r'Implementation-Version: \S+\r', manifest_file_content)[0].split(' ')[1]
        except Exception as error:
            RuleKit._logger.error('Failed to read RuleKit versions from jar file')
            RuleKit._logger.error(error)
            raise error

    @staticmethod
    def _launch_jvm(initial_heap_size: int, max_heap_size: int):
        if jpype.isJVMStarted():
            RuleKit._logger.info('JVM already running')
        else:
            params = [
                f'-Djava.class.path={RuleKit._class_path}',
            ]
            if initial_heap_size is not None:
                params.append(f'-Xms{initial_heap_size}m')
            if max_heap_size is not None:
                params.append(f'-Xmx{max_heap_size}m')
            jpype.startJVM(jpype.getDefaultJVMPath(), *params, convertStrings=False)
