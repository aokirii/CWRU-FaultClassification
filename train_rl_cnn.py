# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, Model # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
from tqdm import tqdm
import math
import time
import os
import scipy.io
from scipy import signal as sp_signal
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

CWRU_DATA_PATH = '/Users/kasimesen/Desktop/Kodlar/Python/12KDriveEndBearing'
SEQUENCE_LENGTH = 1024
STFT_NPERSEG = 24
STFT_NOVERLAP = 12
STFT_NFFT = 128
FREQ_BINS = STFT_NFFT // 2 + 1
TIME_BINS = (SEQUENCE_LENGTH - STFT_NPERSEG) // (STFT_NPERSEG - STFT_NOVERLAP) + 1
YOUR_INPUT_SHAPE = (FREQ_BINS, TIME_BINS, 1)
YOUR_NUMBER_OF_CLASSES = 10
YOUR_STATE_DIM = 6
YOUR_ACTION_DIM = 5
INITIAL_LEARNING_RATE = 0.01
YOUR_CNN_BATCH_SIZE = 40
N_FOLDS = 5
LOAD_CONDITIONS = [0, 1, 2, 3]

BUFFER_SIZE = 35000
BATCH_SIZE_DDQN = 50
GAMMA = 0.99
Q_LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000
Q_NET_ARCH = [16, 16]
RESCALE_RANGE = [0.1, 0.9]
EPSILON = 0.3
FLOAT32_MAX = np.finfo(np.float32).max / 10
FLOAT32_MIN = np.finfo(np.float32).min / 10
CLIP_RANGE = 1e9

class QNetwork(Model):
    def __init__(self, state_dim: int = YOUR_STATE_DIM, action_dim: int = YOUR_ACTION_DIM, hidden_units: list = Q_NET_ARCH, **kwargs):
        super().__init__(name='QNetwork', **kwargs)
        self.state_dim = state_dim; self.action_dim = action_dim; self.hidden_units = hidden_units
        self.internal_layers_list = []
        for units in self.hidden_units: self.internal_layers_list.append(layers.Dense(units, activation='relu', kernel_initializer='he_normal'))
        self.output_layer = layers.Dense(self.action_dim, activation='linear')
    def build(self, input_shape): super().build(input_shape); self.built = True
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if not isinstance(x, tf.Tensor): x = tf.convert_to_tensor(x, dtype=tf.float32)
        elif x.dtype != tf.float32: x = tf.cast(x, tf.float32)
        for layer in self.internal_layers_list: x = layer(x, training=training)
        x = self.output_layer(x, training=training); return x

class Rescaler:
    def __init__(self, low: float = RESCALE_RANGE[0], high: float = RESCALE_RANGE[1], epsilon: float = 1e-6):
        self.low = low; self.high = high; self.range = high - low; self.epsilon = epsilon
        self.running_min = tf.Variable(np.inf, dtype=tf.float32, trainable=False); self.running_max = tf.Variable(-np.inf, dtype=tf.float32, trainable=False); self.alpha = 0.01
    def fit(self, y):
        if not isinstance(y, tf.Tensor): y = tf.convert_to_tensor(y, dtype=tf.float32)
        finite_y = y[tf.math.is_finite(y)];
        if tf.size(finite_y) == 0: return
        mn = tf.reduce_min(finite_y); mx = tf.reduce_max(finite_y)
        if not tf.math.is_finite(self.running_min) or not tf.math.is_finite(self.running_max):
             self.running_min.assign(tf.clip_by_value(mn, FLOAT32_MIN, FLOAT32_MAX))
             self.running_max.assign(tf.clip_by_value(mx, FLOAT32_MIN, FLOAT32_MAX))
        else:
             new_min = self.alpha * mn + (1.0 - self.alpha) * self.running_min
             new_max = self.alpha * mx + (1.0 - self.alpha) * self.running_max
             self.running_min.assign(tf.clip_by_value(new_min, FLOAT32_MIN, FLOAT32_MAX))
             self.running_max.assign(tf.clip_by_value(new_max, FLOAT32_MIN, FLOAT32_MAX))
        if self.running_min > self.running_max:
            self.running_min.assign(tf.clip_by_value(mn, FLOAT32_MIN, FLOAT32_MAX))
            self.running_max.assign(tf.clip_by_value(mx, FLOAT32_MIN, FLOAT32_MAX))
    def scale(self, y):
        if not isinstance(y, tf.Tensor): y = tf.convert_to_tensor(y, dtype=tf.float32)
        run_min = self.running_min.numpy(); run_max = self.running_max.numpy()
        if not np.isfinite(run_min) or not np.isfinite(run_max): return tf.fill(tf.shape(y), (self.low + self.high) / 2.0)
        rng = run_max - run_min; rng = np.clip(rng, self.epsilon, CLIP_RANGE)
        if rng < self.epsilon: return tf.fill(tf.shape(y), (self.low + self.high) / 2.0)
        finite_mask = tf.math.is_finite(y); y_np = y.numpy(); finite_mask_np = finite_mask.numpy(); scaled_vals = np.zeros_like(y_np)
        y_masked = y_np[finite_mask_np]; term = (y_masked - run_min) / rng; term = np.clip(term, -CLIP_RANGE, CLIP_RANGE)
        scaled_vals_masked = self.low + self.range * term; scaled_vals_masked = np.clip(scaled_vals_masked, FLOAT32_MIN, FLOAT32_MAX)
        scaled_vals[finite_mask_np] = scaled_vals_masked
        scaled_vals[~np.isfinite(scaled_vals)] = (self.low + self.high) / 2.0; clipped_vals = np.clip(scaled_vals, self.low, self.high); output = np.where(finite_mask_np, clipped_vals, (self.low + self.high) / 2.0)
        return tf.convert_to_tensor(output, dtype=tf.float32)
    def inv_scale(self, y):
        if not isinstance(y, tf.Tensor): y = tf.convert_to_tensor(y, dtype=tf.float32)
        run_min = self.running_min.numpy(); run_max = self.running_max.numpy()
        if not np.isfinite(run_min) or not np.isfinite(run_max): return y
        rng = run_max - run_min; rng = np.clip(rng, self.epsilon, CLIP_RANGE)
        if rng < self.epsilon: return tf.fill(tf.shape(y), run_min)
        finite_mask = tf.math.is_finite(y); y_np = y.numpy(); finite_mask_np = finite_mask.numpy(); inv_scaled_vals = np.zeros_like(y_np)
        y_masked = y_np[finite_mask_np]; term = (y_masked - self.low) * rng / self.range; term = np.clip(term, -CLIP_RANGE, CLIP_RANGE)
        inv_scaled_vals_masked = run_min + term; inv_scaled_vals_masked = np.clip(inv_scaled_vals_masked, FLOAT32_MIN, FLOAT32_MAX)
        inv_scaled_vals[finite_mask_np] = inv_scaled_vals_masked
        inv_scaled_vals[~np.isfinite(inv_scaled_vals)] = run_min; output = np.where(finite_mask_np, inv_scaled_vals, run_min)
        return tf.convert_to_tensor(output, dtype=tf.float32)

class DDQNAgent:
    def __init__( self, state_dim: int = YOUR_STATE_DIM, action_dim: int = YOUR_ACTION_DIM, buffer_size: int = BUFFER_SIZE, batch_size: int = BATCH_SIZE_DDQN, gamma: float = GAMMA, q_learning_rate: float = Q_LEARNING_RATE, target_update_freq: int = TARGET_UPDATE_FREQ ):
        self.state_dim = state_dim; self.action_dim = action_dim; self.buffer_size = buffer_size; self.batch_size = batch_size; self.gamma = tf.constant(gamma, dtype=tf.float32); self.target_update_freq = target_update_freq
        self.main_net = QNetwork(state_dim, action_dim); self.target_net = QNetwork(state_dim, action_dim);
        self.main_net.build((None, state_dim)); self.target_net.build((None, state_dim)); self.target_net.set_weights(self.main_net.get_weights())
        self.optimizer = optimizers.Adam(learning_rate=q_learning_rate); self.replay_buffer = deque(maxlen=buffer_size); self.rescaler = Rescaler(); self.loss_fn = losses.MeanSquaredError()
        self._update_counter = tf.Variable(0, dtype=tf.int64, trainable=False, name="update_counter");
    def store_transition(self, state, action, reward, next_state, done): state = np.asarray(state, dtype=np.float32); next_state = np.asarray(next_state, dtype=np.float32); transition = (state, int(action), float(reward), next_state, bool(done)); self.replay_buffer.append(transition)
    def sample_batch(self):
        if len(self.replay_buffer) < self.batch_size: return None
        batch = random.sample(self.replay_buffer, self.batch_size); s, a, r, ns, d = map(np.array, zip(*batch))
        return (tf.convert_to_tensor(s, dtype=tf.float32), tf.convert_to_tensor(a, dtype=tf.int32), tf.convert_to_tensor(r, dtype=tf.float32), tf.convert_to_tensor(ns, dtype=tf.float32), tf.convert_to_tensor(d, dtype=tf.bool))
    def _train_step(self, states, actions, rewards, next_states, dones):
        q_main_next = self.main_net(next_states, training=False); best_next_actions = tf.argmax(q_main_next, axis=1, output_type=tf.int32); indices = tf.stack([tf.range(tf.shape(best_next_actions)[0], dtype=tf.int32), best_next_actions], axis=1)
        q_target_next = tf.gather_nd(self.target_net(next_states, training=False), indices)
        inv_q_target_next = self.rescaler.inv_scale(q_target_next)
        rewards = tf.reshape(rewards, [-1]); dones_float = tf.cast(tf.reshape(dones, [-1]), tf.float32); target_q_values_orig = rewards + self.gamma * inv_q_target_next * (1.0 - dones_float)
        target_q_values_scaled = self.rescaler.scale(target_q_values_orig)
        target_q_values_scaled = tf.reshape(target_q_values_scaled, [-1])
        with tf.GradientTape() as tape:
            all_q_main_current = self.main_net(states, training=True); action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            q_main_current = tf.gather_nd(all_q_main_current, action_indices); loss = self.loss_fn(target_q_values_scaled, q_main_current)
        grads = tape.gradient(loss, self.main_net.trainable_variables); valid_grads_vars = [(g, v) for g, v in zip(grads, self.main_net.trainable_variables) if g is not None]
        if valid_grads_vars: self.optimizer.apply_gradients(valid_grads_vars)
        return loss, target_q_values_orig
    def update(self):
        batch = self.sample_batch();
        if batch is None: return None
        states, actions, rewards, next_states, dones = batch;
        loss, target_q_values_orig = self._train_step(states, actions, rewards, next_states, dones)
        self.rescaler.fit(target_q_values_orig); self._update_counter.assign_add(1)
        if self._update_counter % self.target_update_freq == 0: self.target_net.set_weights(self.main_net.get_weights())
        return float(loss.numpy())
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, YOUR_STATE_DIM), dtype=tf.float32)])
    def get_q_values(self, state_tensor): return self.main_net(state_tensor, training=False)
    def get_action(self, state: np.ndarray, policy) -> int: state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32); q_values = self.get_q_values(state_tensor); action = policy.select_action(q_values, self._update_counter); return action

class EpsilonGreedyPolicy:
    def __init__(self, action_dim: int = YOUR_ACTION_DIM, epsilon: float = EPSILON):
        self.action_dim = action_dim
        self.epsilon = epsilon
    def select_action(self, q_values: tf.Tensor, current_step: tf.Variable) -> int:
        epsilon = self.epsilon
        if tf.random.uniform(()) < epsilon: action = tf.random.uniform(shape=(), minval=0, maxval=self.action_dim, dtype=tf.int32)
        else: action = tf.argmax(q_values, axis=1, output_type=tf.int32)[0]
        return int(action.numpy())

class RL_Env:
    def __init__(self, initial_lr: float, action_dim: int = YOUR_ACTION_DIM, state_dim: int = YOUR_STATE_DIM):
        self.current_lr = tf.Variable(initial_lr, dtype=tf.float32, trainable=False); self.action_dim = action_dim; self.state_dim = state_dim
        self.alpha1 = 0.99; self.alpha2 = 0.996
        self.previous_loss = tf.Variable(np.inf, dtype=tf.float32); self.previous_grads_norm = tf.Variable(0.0, dtype=tf.float32); self.iteration_count = tf.Variable(0, dtype=tf.int64)
        self.lowest_losses = tf.Variable(tf.fill([5], np.inf), dtype=tf.float32); self.previous_grad_signs = None;
    def reset(self): self.previous_loss.assign(np.inf); self.previous_grads_norm.assign(0.0); self.iteration_count.assign(0); self.lowest_losses.assign(tf.fill([tf.shape(self.lowest_losses)[0]], np.inf)); self.previous_grad_signs = None;
    def _calculate_grad_norm(self, grads):
        if not grads: return tf.constant(0.0, dtype=tf.float32)
        valid_grads = [g for g in grads if g is not None];
        if not valid_grads: return tf.constant(0.0, dtype=tf.float32)
        flat_grads = tf.concat([tf.reshape(g, [-1]) for g in valid_grads], axis=0);
        if tf.size(flat_grads) == 0: return tf.constant(0.0, dtype=tf.float32)
        norm = tf.norm(flat_grads); return norm
    def _calculate_state_alignment(self, current_grads):
        if self.previous_grad_signs is None or not current_grads: return tf.constant(0.0, dtype=tf.float32)
        valid_current_grads = [g for g in current_grads if g is not None];
        if not valid_current_grads: return tf.constant(0.0, dtype=tf.float32)
        current_flat_grads = tf.concat([tf.reshape(g, [-1]) for g in valid_current_grads], axis=0)
        if self.previous_grad_signs is not None and tf.shape(current_flat_grads)[0] == tf.shape(self.previous_grad_signs)[0]:
             current_signs = tf.sign(current_flat_grads); alignment = tf.reduce_mean(tf.cast(current_signs * self.previous_grad_signs > 0, tf.float32)); return alignment
        else: return tf.constant(0.0, dtype=tf.float32)
    def calculate_state(self, current_loss: tf.Tensor, current_grads: list) -> np.ndarray:
        current_loss_val = tf.cast(current_loss, tf.float32); current_grads_norm = self._calculate_grad_norm(current_grads); state_lr = self.current_lr; state_loss = current_loss_val; state_grad_norm = current_grads_norm
        state_iter = tf.cast(self.iteration_count, tf.float32) / 10000.0; min_hist_loss = tf.reduce_min(self.lowest_losses); finite_losses = tf.boolean_mask(self.lowest_losses, tf.math.is_finite(self.lowest_losses))
        max_hist_loss = tf.reduce_max(finite_losses) if tf.size(finite_losses) > 0 else -np.inf;
        if not tf.math.is_finite(min_hist_loss): min_hist_loss = np.inf
        if current_loss_val <= min_hist_loss: state_min_max = 1.0
        elif min_hist_loss < current_loss_val <= max_hist_loss: state_min_max = 0.0
        else: state_min_max = -1.0
        state_alignment = self._calculate_state_alignment(current_grads); state_vector = tf.stack([state_lr, state_loss, state_grad_norm, state_iter, state_min_max, state_alignment], axis=0)
        self.previous_loss.assign(current_loss_val); self.previous_grads_norm.assign(current_grads_norm); self.iteration_count.assign_add(1)
        losses_concat = tf.concat([self.lowest_losses, [current_loss_val]], axis=0); finite_losses_concat = tf.boolean_mask(losses_concat, tf.math.is_finite(losses_concat)); sorted_losses = tf.sort(finite_losses_concat)
        num_to_keep = tf.shape(self.lowest_losses)[0]
        if tf.size(sorted_losses) >= num_to_keep: self.lowest_losses.assign(sorted_losses[:num_to_keep])
        else: padding = tf.fill([num_to_keep - tf.size(sorted_losses)], np.inf); self.lowest_losses.assign(tf.concat([sorted_losses, padding], axis=0))
        valid_current_grads = [g for g in current_grads if g is not None]
        if valid_current_grads:
             flat_grads = tf.concat([tf.reshape(g, [-1]) for g in valid_current_grads], axis=0)
             if tf.size(flat_grads) > 0: self.previous_grad_signs = tf.sign(flat_grads)
        state_vector = tf.where(tf.math.is_finite(state_vector), state_vector, tf.zeros_like(state_vector))
        if state_vector is None: return np.zeros(self.state_dim, dtype=np.float32)
        return state_vector.numpy()
    def apply_action_and_get_reward(self, action: int, current_loss: tf.Tensor) -> tuple[tf.Tensor, float]:
        lr_before = self.current_lr.value(); new_lr = self.current_lr
        if action == 0: new_lr = self.current_lr / self.alpha1
        elif action == 1: new_lr = self.current_lr / self.alpha2
        elif action == 3: new_lr = self.current_lr * self.alpha2
        elif action == 4: new_lr = self.current_lr * self.alpha1
        new_lr_clipped = tf.clip_by_value(new_lr, 1e-7, 1.0); self.current_lr.assign(new_lr_clipped)
        loss_float = float(current_loss.numpy()) if tf.is_tensor(current_loss) else float(current_loss); reward = 1.0 / (loss_float + 1e-9) if np.isfinite(loss_float) else 0.0
        reward = np.clip(reward, -1000.0, 1000.0); return self.current_lr, float(reward)
    def get_current_lr(self) -> tf.Tensor: return self.current_lr

def cnn_train_step(cnn_model, optimizer, loss_fn, x_batch, y_batch):
    with tf.GradientTape() as tape:
        preds = cnn_model(x_batch, training=True)
        loss = loss_fn(y_batch, preds)
        if cnn_model.losses: loss += tf.add_n(cnn_model.losses)
    grads = tape.gradient(loss, cnn_model.trainable_variables)
    valid_grads_vars = [(g, v) for g, v in zip(grads, cnn_model.trainable_variables) if g is not None]
    if valid_grads_vars: optimizer.apply_gradients(valid_grads_vars)
    valid_grads = [g for g, _ in valid_grads_vars]; return loss, valid_grads

def run_rl_controlled_cnn_step(cnn_model, main_optimizer, loss_fn, rl_env, ddqn_agent, policy, current_state, x_batch, y_batch, store_transition=True):
    if current_state is None or not isinstance(current_state, np.ndarray): raise ValueError(f"Invalid state type: {type(current_state)}. Expected numpy.ndarray.")
    if current_state.shape == (0,) or (len(current_state.shape) > 0 and current_state.shape[0] != rl_env.state_dim): raise ValueError(f"Invalid state dimension: {current_state.shape}. Expected ({rl_env.state_dim},)")
    action = ddqn_agent.get_action(current_state, policy); lr_prev = rl_env.get_current_lr(); main_optimizer.learning_rate.assign(lr_prev)
    loss_cnn, grads = cnn_train_step(cnn_model, main_optimizer, loss_fn, x_batch, y_batch)
    new_lr, reward = rl_env.apply_action_and_get_reward(action, loss_cnn); main_optimizer.learning_rate.assign(new_lr)
    next_state = rl_env.calculate_state(loss_cnn, grads)
    if store_transition:
        if np.any(np.isnan(current_state)) or np.any(np.isnan(next_state)): print("Warning: NaN detected in state/next_state. Skipping transition storage.")
        else: ddqn_agent.store_transition(current_state, action, reward, next_state, False)
    return next_state, reward, loss_cnn, lr_prev

def train_cnn_with_rl(main_cnn, game_cnn, main_optimizer, game_optimizer, loss_fn, ddqn_agent, rl_env, data_loader, config):
    epochs = config.get('epochs', 10); step4game = config.get('step4game', 5); policy = EpsilonGreedyPolicy()
    history = {'epoch': [], 'step': [], 'main_cnn_loss': [], 'game_cnn_loss': [],
               'q_loss': [], 'reward': [], 'lr': [],
               'game_cnn_time': [], 'ddqn_time': [], 'main_cnn_time': []}
    training_steps = 0; state = None; start_time_total = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        sum_main_loss = sum_game_loss = sum_q_loss = sum_reward = 0.0
        sum_game_cnn_time = sum_ddqn_time = sum_main_cnn_time = 0.0
        q_updates_count = batch_count = 0
        pbar_epoch = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for x_batch, y_batch in pbar_epoch:
            x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
            if y_batch.dtype != tf.int32 and y_batch.dtype != tf.int64: y_batch = tf.cast(y_batch, tf.int32)
            game_cnn.set_weights(main_cnn.get_weights())
            if state is None:
                rl_env.reset(); initial_lr_val = rl_env.get_current_lr(); main_optimizer.learning_rate.assign(initial_lr_val); game_optimizer.learning_rate.assign(initial_lr_val)
                loss0, grads0 = cnn_train_step(main_cnn, main_optimizer, loss_fn, x_batch, y_batch)
                if tf.math.is_finite(loss0): state = rl_env.calculate_state(loss0, grads0);
                else: print("Warning: Initial loss is not finite. Skipping batch..."); continue

            time_game_start = time.time()
            game_state = state.copy(); current_game_loss_sum = 0.0
            for i_game in range(step4game):
                action = ddqn_agent.get_action(game_state, policy); current_lr_game = rl_env.get_current_lr(); temp_lr = current_lr_game
                if action == 0: temp_lr = temp_lr / rl_env.alpha1
                elif action == 1: temp_lr = temp_lr / rl_env.alpha2
                elif action == 3: temp_lr = temp_lr * rl_env.alpha2
                elif action == 4: temp_lr = temp_lr * rl_env.alpha1
                temp_lr = tf.clip_by_value(temp_lr, 1e-7, 1.0); game_optimizer.learning_rate.assign(temp_lr)
                loss_game, grads_game = cnn_train_step(game_cnn, game_optimizer, loss_fn, x_batch, y_batch)
                if tf.math.is_finite(loss_game):
                    current_game_loss_sum += float(loss_game.numpy()); reward_game = 1.0 / (float(loss_game.numpy()) + 1e-9); reward_game = np.clip(reward_game, -1000.0, 1000.0)
                    next_game_state = rl_env.calculate_state(loss_game, grads_game)
                    if not (np.any(np.isnan(game_state)) or np.any(np.isnan(next_game_state))): ddqn_agent.store_transition(game_state, action, reward_game, next_game_state, False)
                    game_state = next_game_state
            avg_game_loss_step = current_game_loss_sum / step4game if step4game > 0 else 0.0; sum_game_loss += avg_game_loss_step
            time_game_end = time.time()
            sum_game_cnn_time += (time_game_end - time_game_start)

            time_ddqn_start = time.time()
            q_loss = ddqn_agent.update()
            time_ddqn_end = time.time()
            sum_ddqn_time += (time_ddqn_end - time_ddqn_start)
            if q_loss is not None: sum_q_loss += q_loss; q_updates_count += 1

            time_main_start = time.time()
            main_state = state.copy(); current_main_loss_sum = 0.0; current_reward_sum = 0.0; last_lr_in_step = rl_env.get_current_lr()
            for i_main in range(step4game):
                try:
                    next_main_state, reward, loss_main, lr_step = run_rl_controlled_cnn_step( main_cnn, main_optimizer, loss_fn, rl_env, ddqn_agent, policy, main_state, x_batch, y_batch, store_transition=True )
                    if tf.math.is_finite(loss_main): current_main_loss_sum += float(loss_main.numpy()); current_reward_sum += reward; main_state = next_main_state; last_lr_in_step = lr_step
                    else: print(f"Warning: Non-finite loss encountered in Main-CNN step {i_main} (LR={main_optimizer.learning_rate.numpy():.6f}). Skipping main step update."); break
                except ValueError as e: print(f"Error during RL controlled step {i_main}: {e}"); break
            time_main_end = time.time()
            sum_main_cnn_time += (time_main_end - time_main_start)

            if i_main == step4game - 1: avg_main_loss_step = current_main_loss_sum / step4game if step4game > 0 else 0.0; avg_reward_step = current_reward_sum / step4game if step4game > 0 else 0.0; sum_main_loss += avg_main_loss_step; sum_reward += avg_reward_step; state = main_state
            else: avg_main_loss_step = np.nan; avg_reward_step = np.nan; state = None
            batch_count += 1; training_steps += 1
            pbar_epoch.set_postfix({'main_loss': f"{avg_main_loss_step:.3f}", 'q_loss': f"{q_loss if q_loss is not None else 0:.3f}", 'reward': f"{avg_reward_step:.3f}", 'lr': f"{last_lr_in_step.numpy():.6f}"})
            if state is None: print("State became invalid, will reset..."); break
        pbar_epoch.close()

        avg_game_cnn_time_epoch = sum_game_cnn_time / batch_count if batch_count else 0
        avg_ddqn_time_epoch = sum_ddqn_time / batch_count if batch_count else 0
        avg_main_cnn_time_epoch = sum_main_cnn_time / batch_count if batch_count else 0

        avg_main_epoch = sum_main_loss / batch_count if batch_count else 0; avg_game_epoch = sum_game_loss / batch_count if batch_count else 0; avg_q_epoch = sum_q_loss / q_updates_count if q_updates_count else 0; avg_reward_epoch = sum_reward / batch_count if batch_count else 0; final_lr_epoch = rl_env.get_current_lr().numpy()
        history['epoch'].append(epoch + 1); history['step'].append(training_steps); history['main_cnn_loss'].append(avg_main_epoch); history['game_cnn_loss'].append(avg_game_epoch); history['q_loss'].append(avg_q_epoch); history['reward'].append(avg_reward_epoch); history['lr'].append(final_lr_epoch)
        history['game_cnn_time'].append(avg_game_cnn_time_epoch)
        history['ddqn_time'].append(avg_ddqn_time_epoch)
        history['main_cnn_time'].append(avg_main_cnn_time_epoch)

        epoch_time = time.time() - epoch_start_time;
        if state is None: break
    total_time = time.time() - start_time_total;
    return main_cnn, ddqn_agent, history

def plot_rl_training_history(history, title_suffix=""):
    required_keys = ['step', 'main_cnn_loss', 'game_cnn_loss', 'q_loss', 'reward', 'lr']
    if not all(key in history for key in required_keys): missing_keys = [key for key in required_keys if key not in history]; print(f"Hata: history sözlüğünde eksik anahtarlar var: {missing_keys}"); return
    if not history['step']: print("Hata: history sözlüğünde çizilecek veri yok (boş)."); return
    steps = history['step']; fig, axs = plt.subplots(3, 2, figsize=(14, 12)); fig.suptitle(f'RL Kontrollü CNN Eğitim Metrikleri{title_suffix}', fontsize=16); fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    axs[0, 0].plot(steps, history['main_cnn_loss'], label='Ana CNN Kaybı', color='tab:blue'); axs[0, 0].set_title('Ana Model Kaybı'); axs[0, 0].set_xlabel('Toplam Eğitim Adımı (Step)'); axs[0, 0].set_ylabel('Ortalama Kayıp (Loss)'); axs[0, 0].grid(True, linestyle='--', alpha=0.6); axs[0, 0].legend()
    axs[0, 1].plot(steps, history['game_cnn_loss'], label='Oyun CNN Kaybı', color='tab:orange'); axs[0, 1].set_title('Oyun Modeli Kaybı'); axs[0, 1].set_xlabel('Toplam Eğitim Adımı (Step)'); axs[0, 1].set_ylabel('Ortalama Kayıp (Loss)'); axs[0, 1].grid(True, linestyle='--', alpha=0.6); axs[0, 1].legend()
    q_loss_vals = np.array(history['q_loss']); valid_q_indices = q_loss_vals > 1e-9; steps_np = np.array(steps)
    if np.any(valid_q_indices): axs[1, 0].plot(steps_np[valid_q_indices], q_loss_vals[valid_q_indices], label='Q-Kaybı (Güncellenenler)', color='tab:green', marker='.', linestyle='-', markersize=3)
    axs[1, 0].plot(steps, q_loss_vals, label='Q-Kaybı (Tümü)', color='tab:gray', alpha=0.5); axs[1, 0].set_title('DDQN Ajanı Q-Kaybı'); axs[1, 0].set_xlabel('Toplam Eğitim Adımı (Step)'); axs[1, 0].set_ylabel('Ortalama Kayıp (Loss)'); axs[1, 0].grid(True, linestyle='--', alpha=0.6); axs[1, 0].legend()
    axs[1, 1].plot(steps, history['reward'], label='Ortalama Ödül', color='tab:red'); axs[1, 1].set_title('Ortalama Adım Başına Ödül'); axs[1, 1].set_xlabel('Toplam Eğitim Adımı (Step)'); axs[1, 1].set_ylabel('Ortalama Ödül (Reward)'); axs[1, 1].grid(True, linestyle='--', alpha=0.6); axs[1, 1].legend()
    axs[2, 0].plot(steps, history['lr'], label='Öğrenme Oranı', color='tab:purple', marker='.', linestyle='-'); axs[2, 0].set_title('Öğrenme Oranı (Epoch Sonu)'); axs[2, 0].set_xlabel('Toplam Eğitim Adımı (Step)'); axs[2, 0].set_ylabel('Öğrenme Oranı'); axs[2, 0].grid(True, linestyle='--', alpha=0.6); axs[2, 0].legend(); axs[2, 0].set_yscale('log')
    axs[2, 1].axis('off'); plt.show()

def apply_stft(segment, nperseg, noverlap, nfft):
    _, _, Zxx = sp_signal.stft(segment, nperseg=nperseg, noverlap=noverlap, nfft=nfft, return_onesided=True)
    stft_magnitude = np.abs(Zxx); stft_log_magnitude = np.log1p(stft_magnitude)
    return stft_log_magnitude

def load_cwru_data(data_path, sequence_length, file_code_mapping,
                   stft_nperseg, stft_noverlap, stft_nfft,
                   drive_end=True, normalize='std'):
    data = []; labels = []; sensor_key_suffix = '_DE_time' if drive_end else '_FE_time'
    label_map = { 'Normal': 0, 'B007': 1, 'B014': 2, 'B021': 3, 'IR007': 4, 'IR014': 5, 'IR021': 6, 'OR007': 7, 'OR014': 8, 'OR021': 9 }
    print(f"CWRU verisi yükleniyor (STFT - Tüm Yükler): Yol={data_path}")
    stft_output_shape = None
    for file_code, filenames in file_code_mapping.items():
        if file_code not in label_map: continue
        current_label = label_map[file_code]
        for filename in filenames:
            file_path = os.path.join(data_path, filename)
            if not os.path.exists(file_path): print(f"Uyarı: Dosya bulunamadı: {file_path}. Atlanıyor."); continue
            try:
                mat_data = scipy.io.loadmat(file_path); found_key = None
                for key in mat_data.keys():
                    if sensor_key_suffix in key: found_key = key; break
                if found_key is None: print(f"Uyarı: '{filename}' dosyasında sensör anahtarı bulunamadı. Atlanıyor."); continue
                signal = mat_data[found_key].flatten(); num_segments = len(signal) // sequence_length
                for i in range(num_segments):
                    segment = signal[i * sequence_length : (i + 1) * sequence_length]
                    stft_result = apply_stft(segment, stft_nperseg, stft_noverlap, stft_nfft)
                    data.append(stft_result); labels.append(current_label)
                    if stft_output_shape is None: stft_output_shape = stft_result.shape; print(f"  STFT çıktı şekli belirlendi: {stft_output_shape}")
            except Exception as e: print(f"Hata: '{filename}' dosyası işlenirken sorun oluştu: {e}")
    if not data: raise ValueError(f"Hiçbir veri yüklenemedi. Mapping ve dosya yolunu kontrol edin.")
    x = np.array(data); y = np.array(labels); num_samples, freq_bins, time_bins = x.shape; x_reshaped = x.reshape(num_samples, -1)
    if normalize == 'std': scaler = StandardScaler(); x_normalized = scaler.fit_transform(x_reshaped); print("Veri StandardScaler ile normalize edildi.")
    elif normalize == 'minmax': scaler = MinMaxScaler(); x_normalized = scaler.fit_transform(x_reshaped); print("Veri MinMaxScaler ile normalize edildi.")
    else: x_normalized = x_reshaped; print("Normalizasyon yapılmadı.")
    x = x_normalized.reshape(num_samples, freq_bins, time_bins, 1)
    print(f"Tüm yükler için veri yüklendi ve işlendi. Toplam örnek: {x.shape[0]}, Şekil: {x.shape[1:]}")
    return x.astype(np.float32), y.astype(np.int32), stft_output_shape

# --- DÜZELTME: Makaledeki CNN yapısı (Son MaxPool dahil) ---
def create_cnn_model(input_shape, num_classes):
    print(f"CNN modeli oluşturuluyor (Makale Yapısı - Tam), giriş şekli: {input_shape}")
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # L1 & L2
        layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # L3 & L4
        layers.Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # L5 & L6
        layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # L7 & L8
        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # L9 & L10
        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # L11 & L12
        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)), # <-- SON MAXPOOL GERİ EKLENDİ
        # FC Layers
        layers.Flatten(),
        layers.Dense(2560, activation='relu'),
        layers.Dense(512, activation='relu'),
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_optimizers(initial_lr):
    main_optimizer = optimizers.SGD(learning_rate=initial_lr, momentum=0.9)
    game_optimizer = optimizers.SGD(learning_rate=initial_lr, momentum=0.9)
    print(f"SGD optimizatörleri oluşturuldu, başlangıç LR: {initial_lr}")
    return main_optimizer, game_optimizer

def create_ddqn_agent():
    return DDQNAgent( state_dim=YOUR_STATE_DIM, action_dim=YOUR_ACTION_DIM, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE_DDQN, gamma=GAMMA, q_learning_rate=Q_LEARNING_RATE, target_update_freq=TARGET_UPDATE_FREQ )

def create_rl_env(initial_lr):
    return RL_Env(initial_lr=initial_lr, action_dim=YOUR_ACTION_DIM, state_dim=YOUR_STATE_DIM)

def plot_all_lr_curves(all_lr_histories, all_steps_histories, num_cols=5):
    num_experiments = len(all_lr_histories)
    if num_experiments == 0: print("Çizdirilecek LR geçmişi bulunamadı."); return
    num_rows = math.ceil(num_experiments / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), sharex=True, sharey=True)
    fig.suptitle('Tüm Deneyler İçin Öğrenme Oranı Eğrileri (Log10 Ölçek)', fontsize=16)
    axs = axs.flatten()
    max_steps = 0
    for steps in all_steps_histories:
        if steps: max_steps = max(max_steps, steps[-1])
    for i in range(num_experiments):
        lr_history = all_lr_histories[i]; step_history = all_steps_histories[i]
        if not lr_history or not step_history: continue
        log_lr = np.log10(np.maximum(np.array(lr_history), 1e-9))
        ax = axs[i]; ax.plot(step_history, log_lr, marker='.', linestyle='-', markersize=2); ax.set_title(f'Deney {i+1}'); ax.grid(True, linestyle='--', alpha=0.6); ax.set_xlim(0, max_steps)
        if i >= num_experiments - num_cols: ax.set_xlabel('Adım (Step)')
        if i % num_cols == 0: ax.set_ylabel('LR (Log10)')
    for j in range(num_experiments, len(axs)): axs[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def plot_average_lr_curve(all_histories):
    if not all_histories: print("Ortalama LR çizdirmek için geçmiş verisi bulunamadı."); return
    valid_histories = [h for h in all_histories if h and h['lr'] and h['epoch']]
    if not valid_histories: print("Ortalama LR çizdirmek için geçerli geçmiş verisi bulunamadı."); return
    min_epochs = min(len(h['epoch']) for h in valid_histories) if valid_histories else 0
    if min_epochs == 0: print("Geçmiş verilerinde hiç epoch bulunamadı."); return
    avg_lrs = []; epochs_axis = list(range(1, min_epochs + 1))
    for epoch_idx in range(min_epochs):
        epoch_lrs = [h['lr'][epoch_idx] for h in valid_histories]; avg_lrs.append(np.mean(epoch_lrs))
    plt.figure(figsize=(10, 5)); plt.plot(epochs_axis, np.log10(np.maximum(avg_lrs, 1e-9)), marker='o', linestyle='-', label='Ortalama LR')
    plt.title(f'Tüm Deneyler İçin Ortalama Öğrenme Oranı ({len(valid_histories)} deney)'); plt.xlabel('Epoch'); plt.ylabel('Ortalama LR (Log10 Ölçek)')
    plt.xticks(epochs_axis); plt.grid(True, linestyle='--', alpha=0.6); plt.legend(); plt.yscale('log'); plt.tight_layout(); plt.show()

def print_time_summary(all_histories, total_training_time):
    if not all_histories: print("Zamanlama özeti için geçmiş verisi bulunamadı."); return
    required_time_keys = ['game_cnn_time', 'ddqn_time', 'main_cnn_time']
    valid_histories = [h for h in all_histories if h and all(key in h for key in required_time_keys)]
    if not valid_histories: print("Zamanlama özeti için geçerli zaman verisi içeren geçmiş bulunamadı."); return
    total_game_cnn_time = sum(sum(h['game_cnn_time']) for h in valid_histories)
    total_ddqn_time = sum(sum(h['ddqn_time']) for h in valid_histories)
    total_main_cnn_time = sum(sum(h['main_cnn_time']) for h in valid_histories)
    measured_total = total_game_cnn_time + total_ddqn_time + total_main_cnn_time
    total_fold_time_approx = total_training_time
    others_time = max(0, total_fold_time_approx - measured_total)
    num_valid_experiments = len(valid_histories)
    avg_game_cnn = total_game_cnn_time / num_valid_experiments
    avg_ddqn = total_ddqn_time / num_valid_experiments
    avg_main_cnn = total_main_cnn_time / num_valid_experiments
    avg_others = others_time / num_valid_experiments
    avg_total = avg_game_cnn + avg_ddqn + avg_main_cnn + avg_others
    perc_game = (avg_game_cnn / avg_total) * 100 if avg_total > 0 else 0
    perc_ddqn = (avg_ddqn / avg_total) * 100 if avg_total > 0 else 0
    perc_main = (avg_main_cnn / avg_total) * 100 if avg_total > 0 else 0
    perc_others = (avg_others / avg_total) * 100 if avg_total > 0 else 0
    print("\n" + "="*30); print("Ortalama Eğitim Süresi Dağılımı (Tüm Deneyler)"); print("-"*30)
    print(f"{'Bileşen':<15} | {'Ort. Süre (s)':<15} | {'Yüzde (%)':<10}"); print("-"*30)
    print(f"{'DDQN Update':<15} | {avg_ddqn:<15.1f} | {perc_ddqn:<10.1f}")
    print(f"{'Game-CNN Steps':<15} | {avg_game_cnn:<15.1f} | {perc_game:<10.1f}")
    print(f"{'Main-CNN Steps':<15} | {avg_main_cnn:<15.1f} | {perc_main:<10.1f}")
    print(f"{'Diğer İşlemler':<15} | {avg_others:<15.1f} | {perc_others:<10.1f}")
    print("-"*30); print(f"{'Toplam (Ort.)':<15} | {avg_total:<15.1f} | {100.0:<10.1f}"); print("="*30)


if __name__ == "__main__":
    print("TensorFlow Sürümü:", tf.__version__)
    print("GPU Kullanılabilir mi:", tf.config.list_physical_devices('GPU'))

    cwru_file_mapping = {
        'Normal': ['97.mat', '98.mat', '99.mat', '100.mat'],
        'B007': ['118.mat', '119.mat', '120.mat', '121.mat'],
        'B014': ['185.mat', '186.mat', '187.mat', '188.mat'],
        'B021': ['222.mat', '223.mat', '224.mat', '225.mat'],
        'IR007': ['105.mat', '106.mat', '107.mat', '108.mat'],
        'IR014': ['169.mat', '170.mat', '171.mat', '172.mat'],
        'IR021': ['209.mat', '210.mat', '211.mat', '212.mat'],
        'OR007': ['130.mat', '131.mat', '132.mat', '133.mat'],
        'OR014': ['197.mat', '198.mat', '199.mat', '200.mat'],
        'OR021': ['234.mat', '235.mat', '236.mat', '237.mat']
    }

    all_histories = []
    all_val_accuracies = []
    experiment_count = 0
    total_start_time = time.time()
    actual_input_shape = None
    X_all_loads = None
    y_all_loads = None

    print("Tüm CWRU verisi yükleniyor (mapping'deki tüm dosyalar)...")
    try:
        X_all_loads, y_all_loads, stft_shape = load_cwru_data(
            data_path=CWRU_DATA_PATH, sequence_length=SEQUENCE_LENGTH,
            file_code_mapping=cwru_file_mapping,
            stft_nperseg=STFT_NPERSEG, stft_noverlap=STFT_NOVERLAP, stft_nfft=STFT_NFFT,
            normalize='std'
        )
        actual_input_shape = (stft_shape[0], stft_shape[1], 1)
        if actual_input_shape != YOUR_INPUT_SHAPE:
            print(f"Uyarı: Hesaplanan STFT şekli ({actual_input_shape}) sabit YOUR_INPUT_SHAPE ({YOUR_INPUT_SHAPE}) ile eşleşmiyor. Hesaplanan kullanılacak.")
        else: print(f"STFT girdi şekli doğrulandı: {actual_input_shape}")
        num_loaded_classes = len(np.unique(y_all_loads))
        if num_loaded_classes != YOUR_NUMBER_OF_CLASSES:
             print(f"Uyarı: Yüklenen benzersiz sınıf sayısı ({num_loaded_classes}) beklenen sayıdan ({YOUR_NUMBER_OF_CLASSES}) farklı!")
    except (FileNotFoundError, ValueError) as e:
        print(f"Hata: CWRU verisi yüklenemedi: {e}"); exit()
    except ImportError:
        print("Hata: 'scipy' kütüphanesi bulunamadı. Lütfen 'pip install scipy' ile yükleyin."); exit()

    for load_hp_label in LOAD_CONDITIONS:
        print(f"\n===== Deney Grubu (Formalite Yük Koşulu: {load_hp_label} HP) =====")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + load_hp_label)
        fold_num = 0
        for train_index, val_index in skf.split(X_all_loads, y_all_loads):
            fold_num += 1; experiment_count += 1
            print(f"\n--- Deney {experiment_count}/20: (Yük Etiketi={load_hp_label}HP), Kat={fold_num}/{N_FOLDS} ---")
            x_train_fold, x_val_fold = X_all_loads[train_index], X_all_loads[val_index]; y_train_fold, y_val_fold = y_all_loads[train_index], y_all_loads[val_index]
            print("Modeller, ajan ve ortam yeniden oluşturuluyor...")
            current_input_shape = actual_input_shape if actual_input_shape else YOUR_INPUT_SHAPE
            main_cnn = create_cnn_model(current_input_shape, YOUR_NUMBER_OF_CLASSES);
            main_cnn.summary()
            game_cnn = models.clone_model(main_cnn)
            game_cnn.build((None,) + current_input_shape); game_cnn.set_weights(main_cnn.get_weights())
            main_optimizer, game_optimizer = create_optimizers(INITIAL_LEARNING_RATE)
            ddqn_agent = create_ddqn_agent(); rl_env = create_rl_env(INITIAL_LEARNING_RATE)
            loss_fn_instance = losses.SparseCategoricalCrossentropy()
            train_dataset_fold = tf.data.Dataset.from_tensor_slices((x_train_fold, y_train_fold)).shuffle(buffer_size=len(x_train_fold)).batch(YOUR_CNN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
            data_loader_fold = train_dataset_fold
            config = { 'epochs': 10, 'step4game': 5 }
            fold_start_time = time.time()
            try:
                 trained_main_cnn, _, training_history_fold = train_cnn_with_rl(
                     main_cnn, game_cnn, main_optimizer, game_optimizer, loss_fn_instance, ddqn_agent, rl_env, data_loader_fold, config
                 )
                 fold_end_time = time.time(); print(f"Kat {fold_num} eğitimi tamamlandı. Süre: {fold_end_time - fold_start_time:.2f}s")
                 all_histories.append(training_history_fold)
                 print("Doğrulama için model derleniyor...")
                 trained_main_cnn.compile(optimizer=main_optimizer, loss=loss_fn_instance, metrics=['accuracy'])
                 print("Doğrulama seti üzerinde değerlendirme yapılıyor...")
                 val_loss, val_acc = trained_main_cnn.evaluate(x_val_fold, y_val_fold, verbose=0)
                 all_val_accuracies.append(val_acc); print(f"Kat {fold_num} Doğrulama Sonucu - Kayıp: {val_loss:.4f}, Doğruluk: {val_acc:.4f}")
            except Exception as e:
                 print(f"\nHata: (Yük Etiketi={load_hp_label}HP), Kat={fold_num} eğitimi sırasında hata oluştu: {e}"); import traceback; traceback.print_exc(); all_val_accuracies.append(np.nan)

    total_end_time = time.time()
    print("\n" + "="*30); print("TÜM DENEYLER TAMAMLANDI"); print(f"Toplam Süre: {total_end_time - total_start_time:.2f} saniye"); print(f"Tamamlanan Deney Sayısı: {len(all_val_accuracies)}")
    valid_accuracies = [acc for acc in all_val_accuracies if not np.isnan(acc)]
    if valid_accuracies: mean_accuracy = np.mean(valid_accuracies); std_accuracy = np.std(valid_accuracies); print(f"Ortalama Doğrulama Doğruluğu ({len(valid_accuracies)} geçerli deney üzerinden): {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    else: print("Hiçbir deney başarıyla tamamlanamadı.")

    print("\nTüm deneylerin LR eğrileri çizdiriliyor...")
    all_lr_data = [hist['lr'] for hist in all_histories if hist and 'lr' in hist]
    all_steps_data = [hist['step'] for hist in all_histories if hist and 'step' in hist]
    if all_lr_data and len(all_lr_data) == len(all_steps_data):
        plot_all_lr_curves(all_lr_data, all_steps_data)
    else:
        print("Uyarı: Geçerli LR geçmiş verisi bulunamadı veya adım sayıları eşleşmiyor.")

    print("\nOrtalama LR eğrisi çizdiriliyor...")
    if all_histories:
        plot_average_lr_curve(all_histories)
    else:
        print("Ortalama LR çizdirmek için geçmiş verisi yok.")

    print("\nZamanlama özeti hesaplanıyor...")
    if all_histories:
        print_time_summary(all_histories, total_end_time - total_start_time)
    else:
        print("Zamanlama özeti için geçmiş verisi yok.")

