# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from .resnet import resnet18Z as default_resnet18Z
from .resnet import resnet50Z as default_resnet50Z
from .resnet import resnet34Z as default_resnet34Z

from .resnet_ import resnet18 as default_resnet18
from .resnet_ import resnet50 as default_resnet50
from .resnet_ import resnet34 as default_resnet34

from .Inception_ResNet_v2 import InceptionResNetV2


def resnet18Z(method, *args, **kwargs):
    return default_resnet18Z(*args, **kwargs)

def resnet50Z(method, *args, **kwargs):
    return default_resnet50Z(*args, **kwargs)

def resnet34Z(method, *args, **kwargs):
    return default_resnet34Z(*args, **kwargs)

def resnet18(method, *args, **kwargs):
    return default_resnet18(*args, **kwargs)

def resnet50(method, *args, **kwargs):
    return default_resnet50(*args, **kwargs)

def resnet34(method, *args, **kwargs):
    return default_resnet34(*args, **kwargs)

def inception(method, *args, **kwargs):
    return InceptionResNetV2(*args, **kwargs)

__all__ = ["resnet18Z", "resnet50Z", "resnet34Z", "resnet18", "resnet50", "resnet34", "inception"]
