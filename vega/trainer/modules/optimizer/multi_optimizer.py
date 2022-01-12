#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manage LrScheduler class."""
from collections import OrderedDict
from vega.common import ClassFactory, ClassType
from vega.trainer.modules.lr_schedulers import LrScheduler
from vega.common.config import Config
from .optim import Optimizer
from ..conf.optim import OptimConfig


@ClassFactory.register(ClassType.OPTIMIZER)
class MultiOptimizers(object):
    """Register and call multi-optimizer class."""

    config = OptimConfig()

    def __init__(self, config=None):
        """Initialize."""
        self.is_multi_opt = True
        if config is not None:
            self.config = Config(config)
        self._opts = OrderedDict()

    def __call__(self, model=None, distributed=False):
        """Call Optimizer class."""
        for config in self.config:
            name = config.get('model')
            sub_model = getattr(model, config.get('model'))
            sub_opt = Optimizer(config)(sub_model, distributed)
            sub_lr = None
            sub_loss = None
            if config.get('lr_scheduler'):
                sub_lr = LrScheduler(config=config.get('lr_scheduler'))(sub_opt)
            if config.get('loss'):
                sub_loss = ClassFactory.get_instance(ClassType.LOSS, config.get('loss'))
            self._opts[name] = dict(opt=sub_opt, lr=sub_lr, loss=sub_loss, model=sub_model)
        return self

    def get_opts(self):
        """Get opts."""
        return self._opts.items()
