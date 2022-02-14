import {LabelType} from '../enums/LabelType';
import {ProjectType} from '../enums/ProjectType';

export interface ILabelToolkit {
    labelType: LabelType;
    headerText: string;
    imageSrc: string;
    imageAlt: string;
    projectType: ProjectType;
}

export const LabelToolkitData: ILabelToolkit[] = [
    {
        labelType: LabelType.IMAGE_RECOGNITION,
        headerText: 'Image recognition',
        imageSrc: 'ico/object.png',
        imageAlt: 'object',
        projectType: ProjectType.IMAGE_RECOGNITION,
    },
    {
        labelType: LabelType.POLYGON,
        headerText: 'Polygon',
        imageSrc: 'ico/polygon.png',
        imageAlt: 'polygon',
        projectType: ProjectType.OBJECT_DETECTION,
    },
];