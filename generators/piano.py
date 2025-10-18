"""
Piano Generator

Piano旋律・伴奏生成
- MelodyGenerator: 旋律生成
- CompingGenerator: 伴奏生成
- PianoGenerator: 統合
"""

from typing import List, Optional
import random
from generators.base import (
    InstrumentGeneratorBase,
    NoteEvent,
    Section,
    Chord,
    EmotionProfile,
    Emotion,
    GenerationContext,
)


class MelodyGenerator(InstrumentGeneratorBase):
    """Piano旋律生成"""
    
    def __init__(self):
        super().__init__("piano_melody")
        self.pitch_range = (60, 84)  # C4-C6
    
    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext,
    ) -> List[NoteEvent]:
        """
        旋律生成
        
        Args:
            section: セクション情報
            technique: melody (固定)
            emotion: 感情プロファイル
            context: 生成コンテキスト
        
        Returns:
            旋律ノート
        """
        # 1. コード進行から候補音抽出
        candidate_pitches = self._extract_chord_tones(section.chord_progression)
        
        # 2. Emotion → Melody contour
        contour = self._emotion_to_contour(emotion)
        
        # 3. Rhythm pattern生成
        rhythm = self._generate_rhythm(section.tempo, section.duration)
        
        # 4. Pitch + Rhythm → Melody
        melody = self._combine_pitch_rhythm(
            candidate_pitches,
            rhythm,
            contour,
            section.chord_progression,
        )
        
        # 5. Articulation適用
        melody = self._apply_articulation(melody, emotion)
        
        # 6. Dynamics適用
        melody = self.apply_emotion(melody, emotion)
        
        # 7. Quantize
        melody = self._quantize_timing(melody, resolution=0.125)  # 32nd note
        
        return melody
    
    def _extract_chord_tones(self, chords: List[Chord]) -> List[int]:
        """コード構成音 + テンション音を抽出"""
        tones = []
        for chord in chords:
            chord_tones = chord.get_tones(extensions=True)
            # オクターブ展開（C4-C6範囲）
            for tone in chord_tones:
                for octave in [4, 5, 6]:
                    pitch = tone + octave * 12
                    if self.pitch_range[0] <= pitch <= self.pitch_range[1]:
                        tones.append(pitch)
        return list(set(tones))  # 重複除去
    
    def _emotion_to_contour(self, emotion: EmotionProfile) -> str:
        """Emotion → Melody輪郭"""
        contour_map = {
            Emotion.JOY: "ascending",
            Emotion.SORROW: "descending",
            Emotion.TENSION: "wave_large",
            Emotion.PEACE: "wave_small",
            Emotion.EXCITEMENT: "ascending",
            Emotion.MELANCHOLY: "descending",
        }
        return contour_map.get(emotion.primary, "wave_small")
    
    def _generate_rhythm(self, tempo: float, duration: float) -> List[float]:
        """
        Rhythm pattern生成
        
        Args:
            tempo: BPM
            duration: セクション長（拍）
        
        Returns:
            Note開始時刻のリスト（拍単位）
        """
        if tempo < 80:
            # Slow: 4分音符・8分音符中心
            return self._slow_rhythm(duration)
        elif tempo < 130:
            # Mid: 8分音符・16分音符中心
            return self._mid_rhythm(duration)
        else:
            # Fast: 16分音符・32分音符
            return self._fast_rhythm(duration)
    
    def _slow_rhythm(self, duration: float) -> List[float]:
        """Slow rhythmパターン"""
        times = []
        time = 0.0
        while time < duration:
            times.append(time)
            # Quarter note (1.0) または Half note (2.0)
            time += random.choice([1.0, 2.0])
        return times
    
    def _mid_rhythm(self, duration: float) -> List[float]:
        """Mid rhythmパターン"""
        times = []
        time = 0.0
        while time < duration:
            times.append(time)
            # 8th note (0.5) または Quarter note (1.0)
            time += random.choice([0.5, 1.0])
        return times
    
    def _fast_rhythm(self, duration: float) -> List[float]:
        """Fast rhythmパターン"""
        times = []
        time = 0.0
        while time < duration:
            times.append(time)
            # 16th note (0.25) または 8th note (0.5)
            time += random.choice([0.25, 0.5])
        return times
    
    def _combine_pitch_rhythm(
        self,
        candidate_pitches: List[int],
        rhythm: List[float],
        contour: str,
        chords: List[Chord],
    ) -> List[NoteEvent]:
        """Pitch + Rhythm → Melody"""
        melody = []
        
        # Contour適用
        if contour == "ascending":
            sorted_pitches = sorted(candidate_pitches)
        elif contour == "descending":
            sorted_pitches = sorted(candidate_pitches, reverse=True)
        else:
            sorted_pitches = candidate_pitches
        
        for i, time in enumerate(rhythm):
            # Pitch選択
            if contour in ["ascending", "descending"]:
                pitch = sorted_pitches[i % len(sorted_pitches)]
            else:
                # Wave: ランダムだが隣接音を優先
                if i > 0 and melody:
                    prev_pitch = melody[-1].pitch
                    # 隣接音（±2 semitones）を優先
                    nearby = [p for p in sorted_pitches 
                             if abs(p - prev_pitch) <= 4]
                    pitch = random.choice(nearby if nearby else sorted_pitches)
                else:
                    pitch = random.choice(sorted_pitches)
            
            # Duration決定
            if i < len(rhythm) - 1:
                duration = rhythm[i + 1] - time
            else:
                duration = 1.0  # 最後の音符
            
            # Velocity
            velocity = random.randint(60, 85)
            
            note = NoteEvent(
                pitch=pitch,
                velocity=velocity,
                time=time,
                duration=duration,
            )
            melody.append(note)
        
        return melody
    
    def _apply_articulation(
        self,
        notes: List[NoteEvent],
        emotion: EmotionProfile,
    ) -> List[NoteEvent]:
        """Articulation適用"""
        if emotion.primary in [Emotion.SORROW, Emotion.MELANCHOLY, Emotion.PEACE]:
            # Legato: Note overlap
            for i in range(len(notes) - 1):
                notes[i].duration += 0.1
        elif emotion.primary in [Emotion.EXCITEMENT]:
            # Staccato: Short duration
            for note in notes:
                note.duration *= 0.7
        
        return notes


class CompingGenerator(InstrumentGeneratorBase):
    """Piano伴奏生成"""
    
    def __init__(self):
        super().__init__("piano_comping")
        self.pitch_range = (36, 72)  # C2-C5
    
    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext,
    ) -> List[NoteEvent]:
        """
        伴奏生成
        
        Args:
            section: セクション情報
            technique: pop_comping/ballad/jazz_voicing/arpeggio
            emotion: 感情プロファイル
            context: 生成コンテキスト
        
        Returns:
            伴奏ノート
        """
        # 1. Voicing決定
        voicings = self._select_voicing(technique, section.chord_progression)
        
        # 2. Rhythm pattern
        if technique == "pop_comping":
            comping = self._pop_comping_pattern(voicings, section.tempo, section.duration)
        elif technique == "ballad":
            comping = self._ballad_pattern(voicings, section.tempo, section.duration)
        elif technique == "jazz_voicing":
            comping = self._jazz_voicing_pattern(voicings, section.tempo, section.duration)
        elif technique == "arpeggio":
            comping = self._arpeggio_pattern(voicings, section.tempo, section.duration)
        else:
            comping = self._pop_comping_pattern(voicings, section.tempo, section.duration)
        
        # 3. Dynamics
        comping = self.apply_emotion(comping, emotion)
        
        # 4. Quantize
        comping = self._quantize_timing(comping, resolution=0.125)
        
        return comping
    
    def _select_voicing(
        self,
        technique: str,
        chords: List[Chord],
    ) -> List[List[int]]:
        """Voicing選択"""
        voicings = []
        
        for chord in chords:
            tones = chord.get_tones(extensions=True)
            
            if technique == "jazz_voicing":
                # Drop 2 voicing (Root-7th-3rd-5th in C3-C5)
                voicing = [
                    tones[0] + 48,  # Root (C3)
                    tones[3] + 48 if len(tones) > 3 else tones[2] + 48,  # 7th
                    tones[1] + 60,  # 3rd (C4)
                    tones[2] + 60,  # 5th
                ]
            elif technique == "arpeggio":
                # Arpeggio: Root-3rd-5th-Octave
                voicing = [
                    tones[0] + 48,
                    tones[1] + 48,
                    tones[2] + 48,
                    tones[0] + 60,
                ]
            else:
                # Close voicing (Root-3rd-5th in C3)
                voicing = [t + 48 for t in tones[:3]]
            
            # 音域制限
            voicing = [p for p in voicing 
                      if self.pitch_range[0] <= p <= self.pitch_range[1]]
            voicings.append(voicing)
        
        return voicings
    
    def _pop_comping_pattern(
        self,
        voicings: List[List[int]],
        tempo: float,
        duration: float,
    ) -> List[NoteEvent]:
        """Pop comping pattern"""
        notes = []
        time = 0.0
        
        for voicing in voicings:
            # 8分音符刻み（Offbeat accent）
            for i in range(8):
                if i % 2 == 1:  # Offbeat (2, 4, 6, 8拍目)
                    for pitch in voicing:
                        note = NoteEvent(
                            pitch=pitch,
                            velocity=65,
                            time=time + i * 0.5,
                            duration=0.4,
                        )
                        notes.append(note)
            
            time += 4.0  # 4拍/chord
        
        return notes
    
    def _ballad_pattern(
        self,
        voicings: List[List[int]],
        tempo: float,
        duration: float,
    ) -> List[NoteEvent]:
        """Ballad pattern (Whole note sustain)"""
        notes = []
        time = 0.0
        
        for voicing in voicings:
            for pitch in voicing:
                note = NoteEvent(
                    pitch=pitch,
                    velocity=55,
                    time=time,
                    duration=4.0,  # Whole note
                )
                notes.append(note)
            
            time += 4.0
        
        return notes
    
    def _jazz_voicing_pattern(
        self,
        voicings: List[List[int]],
        tempo: float,
        duration: float,
    ) -> List[NoteEvent]:
        """Jazz voicing pattern (Syncopation)"""
        notes = []
        time = 0.0
        
        for voicing in voicings:
            # Syncopated rhythm
            chord_times = [0.0, 0.75, 2.0, 3.5]  # Syncopation
            
            for chord_time in chord_times:
                for pitch in voicing:
                    note = NoteEvent(
                        pitch=pitch,
                        velocity=60,
                        time=time + chord_time,
                        duration=0.6,
                    )
                    notes.append(note)
            
            time += 4.0
        
        return notes
    
    def _arpeggio_pattern(
        self,
        voicings: List[List[int]],
        tempo: float,
        duration: float,
    ) -> List[NoteEvent]:
        """Arpeggio pattern"""
        notes = []
        time = 0.0
        
        for voicing in voicings:
            # 16分音符でアルペジオ
            for i, pitch in enumerate(voicing):
                note = NoteEvent(
                    pitch=pitch,
                    velocity=65,
                    time=time + i * 0.25,
                    duration=0.2,
                )
                notes.append(note)
            
            time += 4.0
        
        return notes


class PianoGenerator(InstrumentGeneratorBase):
    """Piano統合Generator"""
    
    def __init__(self):
        super().__init__("piano")
        self.melody_gen = MelodyGenerator()
        self.comping_gen = CompingGenerator()
        self.pitch_range = (36, 84)  # C2-C6
    
    def generate(
        self,
        section: Section,
        technique: str,
        emotion: EmotionProfile,
        context: GenerationContext,
    ) -> List[NoteEvent]:
        """
        Piano統合生成（Melody + Comping）
        
        Args:
            section: セクション情報
            technique: pop_comping/ballad/jazz_voicing/arpeggio
            emotion: 感情プロファイル
            context: 生成コンテキスト
        
        Returns:
            Piano全パートノート
        """
        # Melody生成
        melody = self.melody_gen.generate(section, "melody", emotion, context)
        
        # Comping生成
        comping = self.comping_gen.generate(section, technique, emotion, context)
        
        # 音域調整（Melody: C4-C6, Comping: C2-C5）
        melody = self._shift_to_high_register(melody)
        comping = self._shift_to_low_register(comping)
        
        # 統合
        all_notes = melody + comping
        
        return all_notes
    
    def _shift_to_high_register(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """Melodyを高音域に移動"""
        return self._shift_to_octave(notes, target_octave=5)
    
    def _shift_to_low_register(self, notes: List[NoteEvent]) -> List[NoteEvent]:
        """Compingを低音域に移動"""
        return self._shift_to_octave(notes, target_octave=3)
