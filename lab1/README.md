
# SPIVT

## Первая лабораторная

### Структура
./dataset - изображения для тренировки
./data - каскадный классификатор
./learn.py - файл для обучения классификатора на SIFT дескрипторах
./imutils.py - вспомогательные функции
./test.py - для тестирования классификатора на тестовых изображениях
./bof.pkl - ключевые точки изображений разных классов
и другие файлы

### Порядок действий
1) Сформировать список позитивных изображений для каскадного классификатора 
`find ./dataset/good/ -name '*.*' > pos.txt`
2) Сформировать список негативных изображений для каскадного классификатора
`find ./dataset/bad/ -name '*.*' > neg.txt`
3) Создать файл описания изображений
`python make_info.py`
4) Создать коллекцию позитивных в vec формате
`opencv_createsamples -info cars.info -num <колво_изображений> -w 48 -h 24 -vec cars.vec`
5) Посмотреть, для надёжности
`opencv_createsamples -vec cars.vec -w 48 -h 24`
6) Обучение каскада
`opencv_traincascade -data data -vec cars.vec -bg neg.txt -numPos <колво_позитивных> -numNeg <колво_негативных> -numStages 2 -w 48 -h 24 -featureType HAAR -precalcValBufSize 512 -precalcIdxBufSize 512 -mode ALL`