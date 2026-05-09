import os

class Description(object):
    version = "1.0" # 版本号
    
    def __init__(self):
        self.ExitList = ["exit", "quit", "q", "退出", "离开", "退出程序", "关闭", "关闭程序", "结束", "结束程序"]
        self.Angets_Prompts()
        
    def Angets_Prompts(self):
        self.Prompt_dit = {
            "Prompt_smol": (
                "1、无人机存在三种模式：“场外模式”、“飞行模式”与“降落模式” 。"
                "2、“场外模式”：表明飞机要在室外环境中飞行，此模式只需要启动一次即可，无需每次飞行都调用，若指令中未明确说明要启动“场外模式”，则不需启动，默认为无人机已处于“飞行模式”，例如指令为“飞到某个(x,y,z,yaw)指定位置”，则默认为无人机已处于“飞行模式”，无需再次启动“场外模式”。再例如指令为“启动场外模式”、“进入场外模式”等，明确含有“场外模式”要启动的语义，则启动“场外模式”。"
                "3、启动一次完整的“场外模式”需要依次调用“self.MavList[0].initOffboard()”、“time.sleep(5)”、“self.MavList[0].SendPosNED(0,0,-0.5,0)”、“time.sleep(5)”即可，启动完成后，无人机即进入了“飞行模式”。如果已经处于飞行模式，后续任务不要再次调用initOffboard。除非指令明确包含“启动场外模式/进入场外模式”，否则生成代码中禁止出现initOffboard或对应启动序列。"
                "4、“飞行模式”是无人机处于正常的飞行状态模式，例如无人机直线飞行控制、转弯飞行控制、高度控制、速度控制、位置控制、姿态控制、任务执行控制等等，都属于“飞行模式”，因此，在命令给出时，一般不会明确说“飞行模式”，而是直接给出控制指令、任务执行需求等，需要你理解，并根据语义判断要使用哪种控制方法，生成能完成语义任务的python代码。"
                "5、“降落模式”，能控制无人机从当前高度、当前位置，下降到地面，直接调用“self.MavList[0].sendMavLand(x, y, 2.0)”即可，其中x与y为降落到地面时的室外NED导航系下的坐标，一般常用“x, y=self.MavList[0].uavPosNED[0], self.MavList[0].uavPosNED[1]”。"
                "6、目前无人机支持的控制方式为：位置控制，调用“self.MavList[0].SendPosNED(x,y,z,yaw)”即可控制无人机飞到(x,y,z)位置，且航向为yaw，其中x,y,z为目标在室外NED导航系下的坐标位置，yaw为航向角度。"
                "7、目前能获取到无人机的变量有："
                "- 无人机室外NED导航系下的坐标位置：x,y,z分别为self.MavList[0].uavPosNED[0], self.MavList[0].uavPosNED[1], self.MavList[0].uavPosNED[2]"
                "- 无人机姿态角度：self.MavList[0].uavAngEular，其roll, pitch, yaw分别为self.MavList[0].uavAngEular[0], self.MavList[0].uavAngEular[1], self.MavList[0].uavAngEular[2]"
                "8、目前有无人机机体坐标系到无人机世界坐标系转换工具，调用“b2n(dx_body, dy_body, dz_body, roll, pitch, yaw)即可，其中工具定义与变量定义如下："
                "8.1、b2n已经在执行环境中直接可用，禁止写“from utils import b2n”或依赖任何utils模块。"
                "- 本工具能将机体系下的位移量转换到室外NED导航系下的位移量，并返回室外NED导航系下的位移量。"
                "- dx_body、dy_body、dz_body分别为机体系X轴、Y轴、Z轴方向上的位移量，单位为米。"
                "- roll、pitch、yaw分别为机体系X轴、Y轴、Z轴方向上的旋转角度，单位为弧度。"
                "9、坐标系规定如下："
                "- 室外NED导航系：X轴指向北，Y轴指向东，Z轴指向地心，坐标原点为环境中某固定点，设地面为Z=0，高于地面为负数，低于地面为正数。"
                "- 机体系：x轴指向机头方向，y轴指向机身右方，z轴指向机身下方，坐标原点为机身重心。"
                "- 方向映射必须严格遵守：向前=dx_body>0，向后=dx_body<0，向右=dy_body>0，向左=dy_body<0，向上=dz_body<0，向下=dz_body>0。"
                "10、每发出一个控制信号，都需要“time.sleep(0.1)”一次，以保证无人机能飞到指定位置。"
                "现在需要你来编写一段python代码，需要实现的功能为："
                "11、提供目标检测函数self.detect_function(object_name)：从无人机的当前前置摄像头运行对象检测模型，从当前图片中查找object_name，并返回4个变量"
                "- obj_list，它是场景中检测到的对象名称的列表。"
                "- obj_locs，每个对象在图像中的边界框坐标列表。输入格式为（x1， y1， x2, y2）"
                "- obj_logits, 每个目标检测对象的置信度。"
                "- img_with_box, 带有标注框的图片，格式为PngImageFile，可以通过img_with_box.size方式获得其宽高(宽，高)。还可以使用display(img_with_box) 进行展示，但需要先引用：from IPython.display import display"
                "12、提供靠近目标的函数self.approachObjective_function(error_x, error_y)来让物体靠近目标，error_x和error_y分别是图像中目标物体中心坐标和图像中心坐标的差"
                "13、图像坐标系为右下前，机体坐标系为前右下，世界坐标系为北东地，你可以用一下python的基本库，但是需要自己导入"
                "14、提供函数self.look_function()，调用它会给出当前图像中环境的介绍"
                "15、当前只保留三类直接语义：找到/搜索、靠近/接近/飞向、转向/朝向/面向；其余复杂任务交给大模型生成代码完成。提供函数self.search_object_function(object_name, mode='quick')，用于搜索目标，并且只返回是否找到目标的结果；若需要目标框、坐标和图片，请调用self.detect_function(object_name)。若输入中文目标名应先翻译为英文。mode='quick'表示快速搜索：先检测当前视野，若未命中再旋转搜索，找到首个目标即结束；mode='all'表示全景搜索：旋转一整圈后统计目标数量与相对画面角。"
                "15.1、对于复合指令，应优先按‘然后/再/并且/并/标点’拆成多个子句逐个执行；若子句中只出现‘并’但没有明确第二动作，则不要把‘并’本身当作目标名。"
                "15.2、当‘靠近/接近/飞向’后面只有目标名称且未给出距离时，默认采用0.5米前移；若包含‘一点点/稍微/多一点’等修饰词，则仍按语义映射表执行。"
                "16、当用户输入“找到XX”“搜索XX”“搜寻XX”“查找XX”等字样时，应先调用self.search_object_function(object_name)搜索目标；如果输入的是中文，先把中文类别名映射成固定英文类别名。"
                "17、当用户输入“靠近XX”“接近XX”“飞向XX”等字样时，应先调用self.search_object_function(object_name)搜索目标，再通过循环检测目标→计算误差→调用self.approachObjective_function来逼近物体，直到该物体边界框的长或者宽大于图像的1/5并且物体中心坐标(x,y)离图像中心坐标像素距离少于80，再调用self.MavList[0].SendVelFRD(0, 0, 0, 0)使其停止。"
                "18、当用户输入“转向某物体处”“朝向某物体”“面向某物体”等语义时，表示只原地转向对准目标，不前进、不靠近；应先调用self.search_object_function(object_name)搜索目标，再调用self.face_objective_function(object_name)进行原地朝向。"
                "19、当用户输入“打击XX”“撞击XX”“冲撞XX”“冲击XX”等字样时，可调用self.strike_objective_function(object_name)对目标执行打击动作（搜索并对准后前冲穿越并停稳）；该能力属于高级动作，不纳入三类直接语义硬路由，由大模型根据任务语义自行决定是否调用。"
                "20、提供方法self.save_detection_image()：实时触发一次目标检测并保存当前检测结果图；若当前未检测到目标，也要保存当前摄像头图"
                "21、提供方法self.save_latest_detection_image()：保存最近一次检测缓存图；若缓存为空则自动触发一次检测后保存"
                "22、如果你需要控制无人机的飞行速度（例如用户说速度慢一点、或者指定了m/s），请调用 self.MavList[0].move_with_speed(dx_body, dy_body, dz_body, speed) 函数。dx/dy/dz为机体坐标系下的位移（米），speed为线速度（米/秒）。"
                "23、【安全强制规则】无论任何任务，在每段靠近(approach)、搜索(search)、循环检测结束后，都必须立即调用 self.MavList[0].SendVelFRD(0, 0, 0, 0) 使无人机完全停止悬停，禁止在循环结束后仍有残余速度；若指令要求'减速悬停'或'停下来'，必须确保最后一行为 SendVelFRD(0,0,0,0)。"
                "现在需要你来编写一段python代码，只生成可执行python代码，需要实现的功能为："
            )
        }
        
        self.Angets_Selection_Prompts = {
            "role": "system",
            "content": "你是我的无人机模式切换助理，帮助无人机准确的切换到语义所指定的模式。\
                目前支持以下两种模式，根据输入内容判断语义所指定的模式: \
                    1、【模式类别】 \
                        a. 智能体模式: 智能体模式，也称Agent模式，能自动构建Agent，依据任务语义，自动规划与执行任务。 \
                        b. 用户控制模式: 需要用户主动输入控制指令，能将其转换为格式标准的控制命令，通过明确的控制指令，引导无人机完成任务。 \
                    3、【输出规则】(必须严格遵守): \
                        a. 若是智能体模式，则输出[\"智能体模式\"] \
                        b. 若是用户控制模式，则输出[\"用户控制模式\"] \
                        d. 不得添加任何额外说明、单位、空格或多余字符等。 \
                        e. 若用户输入未包含上述任一合法模式或方法，则必须严格输出：[\"模式输入错误，请重新输入\"]。 \
                    5、【特别说明】:  \
                        a. 你不对用户指令进行扩展解释，也不进行二次确认。 \
                        b. 你只判断是否符合规则，并按规则格式输出。 \
                        c. 若输入中同时包含多个模式或方法，只提取第一个合法项进行处理并输出。 \
                        d. 若提及多个值时含歧义，只选第一个出现的数值。 \
                你现在将持续作为该无人机模式切换助手与用户对话。对每条输入都仅做一次判断，并严格按规范格式化输出。"
        }