from .models import Experiment, ExperimentName, FlyWireNeuronId

SUGAR_IDS = (
    FlyWireNeuronId(720575940624963786),
    FlyWireNeuronId(720575940630233916),
    FlyWireNeuronId(720575940637568838),
    FlyWireNeuronId(720575940638202345),
    FlyWireNeuronId(720575940617000768),
    FlyWireNeuronId(720575940630797113),
    FlyWireNeuronId(720575940632889389),
    FlyWireNeuronId(720575940621754367),
    FlyWireNeuronId(720575940621502051),
    FlyWireNeuronId(720575940640649691),
    FlyWireNeuronId(720575940639332736),
    FlyWireNeuronId(720575940616885538),
    FlyWireNeuronId(720575940639198653),
    FlyWireNeuronId(720575940639259967),
    FlyWireNeuronId(720575940617937543),
    FlyWireNeuronId(720575940632425919),
    FlyWireNeuronId(720575940633143833),
    FlyWireNeuronId(720575940612670570),
    FlyWireNeuronId(720575940628853239),
    FlyWireNeuronId(720575940629176663),
    FlyWireNeuronId(720575940611875570),
)
P9_IDS = (FlyWireNeuronId(720575940627652358), FlyWireNeuronId(720575940635872101))

EXPERIMENTS: dict[ExperimentName, Experiment] = {
    'sugar': Experiment('sugar', 0, 0, SUGAR_IDS, (200.0,) * 21),
    'p9': Experiment('p9', 1, 1, P9_IDS, (100.0,) * 2),
    'sugar-silenced': Experiment(
        'sugar-silenced', 3, 0, SUGAR_IDS, (200.0,) * 21, SUGAR_IDS
    ),
    'two-class': Experiment(
        'two-class',
        4,
        4,
        SUGAR_IDS + P9_IDS,
        (200.0,) * 21 + (100.0,) * 2,
        first_class_channels=len(SUGAR_IDS),
    ),
    'silent': Experiment('silent', 5, 5, (), ()),
}
