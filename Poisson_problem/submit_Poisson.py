import sys
sys.path.append('.')
from gpinv_source import gpinv
from phi.tf.flow import *


RES = 8
net = dense_net(RES*RES, RES*RES, [64, 128, 64], activation='tanh')


# net.load_weights('netIG_0_it1413.h5')              # use this to start from a pretrained network


ig_learning_rate = vis.control(0.01)                 # set HIG learning rate
adam_learning_rate = vis.control(3*1e-4)             # set Adam learning rate
update_method = vis.control('Adam', ('HIG', 'Adam')) # set HIG or Adam
sqrt_batch_size = vis.control(8, (1, 32))            # set squared batch size



direct_updates = tf.keras.optimizers.SGD(learning_rate=1, momentum=0)  
adam = tf.keras.optimizers.Adam(learning_rate=lambda: adam_learning_rate)


def physics_and_loss(net_input: CenteredGrid):
    native = math.pack_dims(net_input.values, net_input.shape.spatial, channel('s')).native('batch,s')
    x_predicted = net(native)
    x_predicted = math.unpack_dims(math.wrap(x_predicted, net_input.values.shape.batch, channel('s')), 's', net_input.shape.spatial)
    x_predicted = CenteredGrid(x_predicted, x_gt.extrapolation, bounds=x_gt.bounds)
    y_predicted = field.laplace(x_predicted)
    loss = field.l2_loss(y_predicted - net_input).sum
    return x_predicted, y_predicted, loss


@math.jit_compile
def compute_derivatives(net_input):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(net.trainable_variables)
        x_predicted, y_predicted, loss = physics_and_loss(net_input)
    loss_grad = tape.gradient(loss, y_predicted.values.native(y_predicted.shape))
    jac_list = tape.jacobian(y_predicted.values.native(y_predicted.shape), net.trainable_variables, experimental_use_pfor=True)
    return x_predicted, y_predicted, loss, loss_grad, jac_list

@vis.action
def reset():
    net.load_weights(viewer.scene.subpath('net0.h5'))
    math.seed(0)
    global adam
    adam = tf.keras.optimizers.Adam(learning_rate=1)


@vis.action
def save_model():
    net.save_weights(viewer.scene.subpath(f'net{update_method}_{viewer.reset_count}_it{viewer.steps}.h5'))
    viewer.info(f"Model saved to net{update_method}_{viewer.reset_count}_it{viewer.steps}.h5")


viewer = view('x_gt, y_target, x_predicted, y_predicted, y_diff, x_diff, x_PG, y_PG', namespace=globals(), scene=True, play=True, debug=False, select='batch')
net.save_weights(viewer.scene.subpath('net0.h5'))
math.seed(0)
viewer.info(f"Network parameter count: {parameter_count(net)}")
for _ in viewer.range():
    x_gt = CenteredGrid(Noise(batch(batch=sqrt_batch_size**2)), extrapolation.ZERO, x=RES, y=RES)
    y_target = field.laplace(x_gt)
    x_predicted, y_predicted, loss = physics_and_loss(y_target)

    if update_method == 'HIG':
        x_predicted, _, loss, loss_grad, jac_list = compute_derivatives(y_target)
        flat_jac_list = [tf.reshape(j, (np.prod(j.shape[:y_target.shape.rank]), -1)) for j in jac_list]
        flat_jac = tf.concat(flat_jac_list, axis=1)
        flat_grad = tf.reshape(loss_grad, (-1,))
        "Half-inverting matrix" >> viewer
        inv = gpinv(flat_jac, rcond=1e-5)
        "Computing updates" >> viewer
        ig = tf.tensordot(inv, flat_grad, axes=(1, 0))
        inv_grad_list = []  
        start_index = 0
        for weight in net.trainable_weights:
            end_index = start_index + np.prod(weight.shape)
            inv_grad_weight = np.reshape(ig[start_index:end_index], weight.shape) * ig_learning_rate
            inv_grad_list.append(inv_grad_weight)
            start_index = end_index
        direct_updates.apply_gradients(zip(inv_grad_list, net.trainable_weights))
    elif update_method == 'Adam':
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(net.trainable_variables)
            x_predicted, _, loss = physics_and_loss(y_target)
        grad = tape.gradient(loss, net.trainable_variables)
        adam.apply_gradients(zip(grad, net.trainable_weights))
    viewer.log_scalars(x_distance_l1=math.l1_loss(x_predicted - x_gt).mean)
    viewer.log_scalars(**{f"loss_{update_method}_{viewer.reset_count}": loss / sqrt_batch_size**2})
