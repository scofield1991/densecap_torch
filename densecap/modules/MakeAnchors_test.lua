require 'torch'
require 'cutorch'
require 'nn'

require 'ReshapeBoxFeatures'

require 'ApplyBoxTransform'
require 'MakeAnchors'
require 'MakeBoxes'

local tests = {}
local tester = torch.Tester()


-- Make sure that MakeAnchors + ReshapeBoxFeatures + ApplyBoxTransform
-- computes the same thing as MakeBoxes
local function consistencyFactory(dtype)
  return function()
    local x0, y0, sx, sy = 1, 2, 3, 4
    local N, H, W = 2, 3, 5
    local k = 12

    local anchors = torch.randn(2, k):abs():type(dtype)
    local transforms = torch.randn(1, 4 * k, 14, 14):type(dtype)
    

    local make_boxes = nn.MakeBoxes(x0, y0, sx, sy, anchors):type(dtype)
    local boxes1 = make_boxes:forward(transforms)

    local net = nn.Sequential()
    local concat = nn.ConcatTable()
    local s = nn.Sequential()
    s:add(nn.MakeAnchors(x0, y0, sx, sy, anchors))
    s:add(nn.ReshapeBoxFeatures(k))
    concat:add(s)
    concat:add(nn.ReshapeBoxFeatures(k))
    net:add(concat)
    net:add(nn.ApplyBoxTransform())
    net:type(dtype)

    for i,module in ipairs(net:listModules()) do
      print(module)
    end

    local boxTransform_net = nn.Sequential()
    boxTransform_net:add(nn.ApplyBoxTransform())

    --local boxes2 = net:forward(transforms)
    --print(boxes2:size())

    --print (transforms:size())
    local boxes3 = net:forward(transforms)
    --local boxes4 = boxTransform_net:forward(boxes3)
    --print('asdadadasda'..boxes3:size())
    print('dsfsdfsd', boxes3[1]:size())

    --tester:assertTensorEq(boxes1, boxes2, 1e-6)
  end
end

tests.floatConsistencyTest = consistencyFactory('torch.FloatTensor')
--tests.doubleConsistencyTest = consistencyFactory('torch.DoubleTensor')
--tests.cudaConsistencyTest = consistencyFactory('torch.CudaTensor')



tester:add(tests)
tester:run()
